#!/usr/bin/env/python

import time

import numpy as np
import torch
import torch.nn.functional as F

from typing import Any

from src.utils.utils import negative_sampling, remove_self_loops
from src.utils.losses import structure_reconstruction_loss, kl_divergence, negative_entropy
from src.utils.EarlyStoppingCriterion import EarlyStoppingCriterion
from src.utils.LossScheduler import LossScheduler
from src.models.vgae.HGREstimator import HGREstimator
from src.models.vgae.DemographicVGAE import DemographicVGAE
from src.paths import MODEL_DIR


def train_step(
    vgae: torch.nn.Module,
    hgr_net: torch.nn.Module,
    opt_vgae: torch.optim.Optimizer,
    opt_hgr: torch.optim.Optimizer,
    adj: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    s_true: torch.Tensor,
    pos_index: torch.Tensor,
    neg_index: torch.Tensor,
    train_idx: torch.Tensor,
    lmbda: float,
    beta: float,
    hgr_steps: int
) -> tuple[float, float, float, float, float, float, float]:
    """
    Perform a single training step using Min-Max optimization between VGAE and HGR.

    The process involves two phases:
    1. **Max Phase**: Update the HGR network to maximize the correlation estimate
       between the latent representation $Z$ and the target labels $Y$.
    2. **Min Phase**: Update the VGAE to reconstruct the graph and infer sensitive
       attributes $S$, while simultaneously minimizing the correlation estimate
       found by the HGR network.

    Parameters
    ----------
    vgae : nn.Module
        The Demographic VGAE model.
    hgr_net : nn.Module
        The HGR Estimator network.
    opt_vgae : torch.optim.Optimizer
        Optimizer for VGAE parameters.
    opt_hgr : torch.optim.Optimizer
        Optimizer for HGR Estimator parameters.
    adj : torch.Tensor
        Adjacency matrix of shape [N, N].
    x : torch.Tensor
        Node feature matrix of shape [N, D].
    y : torch.Tensor
        Target labels of shape [N].
    s_true : torch.Tensor
        Ground truth sensitive attributes of shape [N].
    pos_index : torch.Tensor
        Positive edge indices [2, E_pos].
    neg_index : torch.Tensor
        Sampled negative edge indices [2, E_neg].
    train_idx : torch.Tensor
        Mask/Indices for the training set.
    lmbda : float
        Current weight for the HGR fairness penalty.
    beta : float
        Current weight for the KL divergence (annealing).
    hgr_steps : int
        Number of HGR estimator updates per VGAE update.

    Returns
    -------
    tuple[float, float, float, float, float, float, float]
        A tuple containing: (total_loss, hgr_val, recon_x, recon_a, kl_div, neg_entropy, recon_s).
    """
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

    # Latent Space Regularization
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
def evaluate(
    vgae: torch.nn.Module,
    hgr_net: torch.nn.Module,
    adj: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    s_true: torch.Tensor,
    pos_index: torch.Tensor,
    neg_index: torch.Tensor,
    beta: float,
    lmbda: float,
    val_idx: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """
    Evaluate Stage 1 model performance on the validation set.

    Parameters
    ----------
    vgae, hgr_net : nn.Module
        The models to evaluate.
    adj : torch.Tensor
        Adjacency matrix of shape [N, N].
    x : torch.Tensor
        Node feature matrix of shape [N, D].
    y : torch.Tensor
        Target labels of shape [N].
    s_true : torch.Tensor
        Ground truth sensitive attributes of shape [N].
    pos_index : torch.Tensor
        Positive edge indices [2, E_pos].
    neg_index : torch.Tensor
        Sampled negative edge indices [2, E_neg].
    beta : float
        Current loss weights for KL.
    lmbda : float
        Current loss weights for HGR.
    val_idx : torch.Tensor
        Mask/Indices for the validation set.

    Returns
    -------
    tuple[float, float, float, float, float, float, float]
        Validation counterparts to the training metrics.
    """
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


def train_vgae(data: object, args: Any) -> None:
    """
    Orchestrate the Stage 1 training process.

    This function handles:
    1. Data preparation and normalization.
    2. Model and optimizer initialization.
    3. Management of LossSchedulers for KL annealing and HGR weight ramping.
    4. The main training loop with periodic validation and early stopping.
    5. Saving the best VGAE weights and the corresponding HGR estimator.

    Parameters
    ----------
    data : GraphDataset
        The dataset to use.
    args : argparse.args
        Training arguments.
    """
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
