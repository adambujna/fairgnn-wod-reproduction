#!/usr/bin/env/python

import time
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.utils import negative_sampling, remove_self_loops
from src.utils.losses import structure_reconstruction_loss, kl_divergence, negative_entropy
from src.utils.EarlyStoppingCriterion import EarlyStoppingCriterion
from src.utils.LossScheduler import LossScheduler
from src.models.vgae.HGREstimator import HGREstimator
from src.models.vgae.DemographicVGAE import DemographicVGAE
from src.paths import MODEL_DIR


def train_step(vgae, hgr_net, opt_vgae, opt_hgr, adj, x, y, s_true,
               pos_index, neg_index, train_idx, lmbda, beta, hgr_steps):
    # --- Max Phase: Train HGR ---
    for _ in range(hgr_steps):
        vgae.eval()
        hgr_net.train()
        opt_hgr.zero_grad()
        with torch.no_grad():
            _, _, z = vgae.encode(adj, x)

        # Estimate dependency between Z and Y
        est_hgr = hgr_net(z[train_idx], y[train_idx].float().unsqueeze(1))
        loss_hgr = -est_hgr
        loss_hgr.backward()
        opt_hgr.step()

    # --- Min Phase: Train VGAE ---
    vgae.train()
    hgr_net.eval()
    opt_vgae.zero_grad()

    x_hat, pos_a_logits, neg_a_logits, s_logits, mu, logvar, z = vgae(
        adj, x, pos_index, neg_index
    )

    # Reconstruction losses log(P(X | A, S)) + log(P(A | S))
    loss_recon_x = F.mse_loss(x_hat[train_idx], x[train_idx])
    loss_recon_a = structure_reconstruction_loss(pos_a_logits, neg_a_logits)
    # loss log(P(S | Z)) - Q(S | Z)
    loss_recon_s = F.cross_entropy(s_logits[train_idx], s_true[train_idx].long())
    s_probs = F.softmax(s_logits, dim=1)
    neg_ent = negative_entropy(s_probs[train_idx])

    # Regularization
    # KL Divergence log(P(Z)) - log(q(Z | X, A))
    kl_div = kl_divergence(mu, logvar)

    # ELBO + Fairness Regularization, beta does KL annealing to prevent posterior collapse
    neg_elbo = loss_recon_x + loss_recon_a + 10 * loss_recon_s + (beta * kl_div) - neg_ent

    hgr_val = hgr_net(z[train_idx], y[train_idx].float().unsqueeze(1))

    loss_total = neg_elbo + (lmbda * hgr_val)
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(vgae.parameters(), 1.0)
    opt_vgae.step()

    return loss_total.item(), hgr_val.item(), loss_recon_x.item(), loss_recon_a.item(), \
           kl_div.item(), neg_ent.item(), loss_recon_s.item()


@torch.no_grad()
def evaluate(vgae, hgr_net, adj, x, y, s_true, pos_index, neg_index, beta, lmbda, val_idx):
    vgae.eval()
    hgr_net.eval()

    x_hat, pos_a, neg_a, s_logits, mu, logvar, z = vgae(adj, x, pos_index, neg_index)

    val_recon_x = F.mse_loss(x_hat[val_idx], x[val_idx])
    val_recon_a = structure_reconstruction_loss(pos_a, neg_a)

    val_recon_s = F.cross_entropy(s_logits[val_idx], s_true[val_idx].long())

    s_probs = F.softmax(s_logits, dim=1)
    val_neg_ent = negative_entropy(s_probs[val_idx])
    val_kl = kl_divergence(mu[val_idx], logvar[val_idx])

    val_hgr = hgr_net(z[val_idx], y[val_idx].float().unsqueeze(1))

    # Total validation metric (proxy for ELBO + Supervision)
    total_val_loss = val_recon_x + val_recon_a + (beta * val_kl) + val_recon_s - val_neg_ent + (lmbda * val_hgr)

    return total_val_loss, val_recon_x, val_recon_a, val_kl, val_recon_s, val_neg_ent, val_hgr


def train_vgae(data, args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # Prep Data
    adj = data.get_adj_matrix(norm="both").to(device)
    edge_index = data.edge_index.to(device)
    # load and normalize input features
    x = data.x.to(device)
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    y = data.y.to(device)
    s_true = data.sensitive.to(device)
    pos_index = remove_self_loops(edge_index)

    train_idx = data.train_idx.to(device)
    val_idx = data.val_idx.to(device)

    # Initialize Models
    vgae = DemographicVGAE(
        input_size=data.x.shape[1],
        hidden_size=args.hidden_vgae,
        latent_size=args.latent_size,
    ).to(device)

    hgr_net = HGREstimator(args.latent_size, 1, hidden_dim=args.hidden_hgr).to(device)

    opt_vgae = torch.optim.Adam(vgae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_hgr = torch.optim.Adam(hgr_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup Saving and Early Stopping
    model_save_basename = f"{MODEL_DIR}/stage1/vgae_{args.dataset}"
    hgr_save_basename = f"{MODEL_DIR}/stage1/hgr_{args.dataset}"
    if args.save_every != -1:    # save initialized weights at epoch 0 also if saving periodically
        torch.save(vgae.state_dict(), model_save_basename + '_0.pt')
        torch.save(hgr_net.state_dict(), hgr_save_basename + '_0.pt')

    # Note: We save both checkpoints in a wrapper or handle hgr_net separately if needed
    early_stop = EarlyStoppingCriterion(
        patience=args.patience,
        best_delta=0.01,
        mode='min',
        start_from_epoch=args.warmup,
        save_basename=model_save_basename
    )
    kl_scheduler = LossScheduler(0.1, int(args.warmup*0.1), args.warmup, 'sigmoid')
    lambda_scheduler = LossScheduler(args.lambda_hgr, int(args.warmup*0.5), args.warmup, 'linear')

    print(f"Starting Stage 1 (VGAE) training on {args.dataset}...")
    t_total = time.time()

    for epoch in range(args.epochs):
        t = time.time()
        neg_index = negative_sampling(edge_index, x.size(0), pos_index.size(1))

        # Inside your training loop:
        beta = kl_scheduler.get_weight(epoch)
        lambda_hgr = lambda_scheduler.get_weight(epoch)

        loss_train, hgr_loss, lx, la, lkl, lent, ls = train_step(
            vgae, hgr_net, opt_vgae, opt_hgr, adj, x, y, s_true,
            pos_index, neg_index, train_idx, lambda_hgr, beta, args.hgr_steps_per_epoch
        )

        # Validate
        val_loss, val_lx, val_la, val_kl, val_ls, val_ent, val_hgr = evaluate(
            vgae, hgr_net, adj, x, y, s_true, pos_index, neg_index, beta, lambda_hgr, val_idx
        )

        if args.print_every != -1 and (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch + 1} | "
                  f"Train Loss: {loss_train:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LS: {ls:.4f} | "
                  f"LX: {lx:.4f} | "
                  f"LA: {la:.4f} | "
                  f"HGR: {hgr_loss:.4f} | "
                  f"KL: {lkl:.4f} | "
                  f"Ent: {lent:.4f} | "
                  f"Time: {time.time() - t:.2f}s")

        if args.save_every != -1 and (epoch + 1) % args.save_every == 0:
            torch.save(vgae.state_dict(), f"{model_save_basename}_{epoch + 1}.pt")
            torch.save(hgr_net.state_dict(), f"{hgr_save_basename}_{epoch + 1}.pt")

        # Early stopping based on reconstruction loss + KL
        if early_stop.step(val_loss, vgae):
            # Also save the HGR net corresponding to the early stoppage epoch
            torch.save(hgr_net.state_dict(), f"{hgr_save_basename}_last.pt")
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if not early_stop.early_stop:   # In case we didn't early stop, save the last hgr anyway
        torch.save(hgr_net.state_dict(), f"{hgr_save_basename}_last.pt")

    print(f"VGAE Training Finished. Total Time: {time.time() - t_total:.2f}s")
