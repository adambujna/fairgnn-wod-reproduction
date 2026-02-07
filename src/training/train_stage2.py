#!/usr/bin/env/python

import time
import os
import numpy as np
import torch
import torch.nn.functional as F

from src.models.fairgnn_wod.FairGNNWOD import FairGNNWOD
from src.models.vgae.DemographicVGAE import DemographicVGAE
from src.utils.losses import prediction_loss, demographic_loss, fairness_loss, independence_loss
from src.utils.metrics import accuracy, f1_score, fairness_metrics, auc_score
from src.utils.utils import format_metric
from src.utils.EarlyStoppingCriterion import EarlyStoppingCriterion
from src.utils.LossScheduler import LossScheduler
from src.paths import MODEL_DIR


def train_discriminator(model, adj, x, edge_index, idx_train):
    model.train()
    model.vgae.eval()

    y_logits, h, c_logits, s_pred = model(adj, x, edge_index, -1, mask=False)

    ld = demographic_loss(c_logits[:, idx_train, :], s_pred[idx_train].long())
    ld.backward()

    return ld.item()


def train_step(model, optimizer, adj, x, y, edge_index, mask, idx_train, adv_steps, alpha, beta, gamma, lambda_):
    model.train()
    model.vgae.eval()
    ld_wu = -0
    for _ in range(adv_steps):
        optimizer.zero_grad()
        ld_wu = train_discriminator(
            model, adj, x, edge_index, idx_train
        )
        optimizer.step()

    # Forward pass: s_pred comes from the internal frozen VGAE
    optimizer.zero_grad()
    y_logits, h, c_logits, s_pred = model(adj, x, edge_index, lambda_, mask)

    # Losses - Stage 2 uses s_pred as a proxy for true demographics
    lp = prediction_loss(y_logits[idx_train], y[idx_train])
    ld = demographic_loss(c_logits[:, idx_train, :], s_pred[idx_train])
    lf = fairness_loss(y_logits[idx_train], s_pred[idx_train])
    li = independence_loss(h[idx_train])

    total_loss = lp + alpha * li + beta * ld + gamma * lf
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), lp.item(), ld.item(), li.item(), lf.item(), ld_wu


def evaluate(model, adj, x, y, edge_index, idx, s_true=None):
    model.eval()
    # Note: FairGNNWOD forward returns (y_logits, h, c_logits, s_probs)
    y_logits, _, _, _ = model(adj, x, edge_index, 0.0, True)

    preds = (y_logits.argmax(dim=-1).squeeze()).long()

    acc = accuracy(preds, y, idx)
    f1 = f1_score(preds, y, idx)

    # AUC
    probs = F.softmax(y_logits, dim=-1)[:, 1].detach().cpu()
    y_cpu = y.detach().cpu()
    idx_cpu = idx.detach().cpu()
    try:
        auc = auc_score(y_cpu[idx_cpu], probs[idx_cpu])
    except ValueError:
        auc = 0.5

    metrics = {"acc": acc, "f1": f1, "auc": auc}

    # Only compute fairness if s_true is provided
    # !!!If you do provide true sensitive attributes, DO NOT use this as a train/val metric
    if s_true is not None:
        sp, eo = fairness_metrics(preds, y, s_true, idx)
        metrics.update({"sp": sp, "eo": eo})

    return metrics


def train_stage2(data, args):
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # Load data
    adj = data.get_adj_matrix(norm='both').to(device)
    x = data.x.to(device)
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    y = data.y.to(device).long()
    edge_index = data.edge_index.to(device)

    idx_train = data.train_idx.to(device)
    idx_val = data.val_idx.to(device)
    idx_test = data.test_idx.to(device)

    # Real demographics
    s_true = data.sensitive.to(device) if data.sensitive is not None else None

    # 1. Load frozen Stage 1 VGAE
    vgae = DemographicVGAE(
        input_size=data.x.shape[1],
        hidden_size=args.hidden_vgae,  # Ensure this matches Stage 1 config
        latent_size=args.latent_size
    ).to(device)

    vgae_path = f"{MODEL_DIR}/stage1/vgae_{args.dataset}_best.pt"
    print(f"Loading pre-trained VGAE from {vgae_path}...")
    vgae.load_state_dict(torch.load(vgae_path, map_location=device))
    vgae.eval()
    for p in vgae.parameters():
        p.requires_grad = False

    # 2. Initialize Stage 2
    model = FairGNNWOD(
        vgae=vgae,
        input_size=data.x.shape[1],
        hidden_size=args.hidden,
        num_channels=args.channels
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    model_save_basename = f"{MODEL_DIR}/stage2/fairgnnwod_{args.dataset}"
    best_model_path = f'{model_save_basename}_best.pt'
    if os.path.exists(best_model_path):
        print(f"Found existing best model at {best_model_path}. Skipping training.")
    else:
        # 3. Setup Saving and Early Stopping
        if args.save_every != -1:    # save initialized weights at epoch 0 also if saving periodically
            torch.save(model.state_dict(), model_save_basename + '_0.pt')
        early_stop = EarlyStoppingCriterion(
            patience=args.patience,
            mode='max',
            start_from_epoch=args.warmup,
            save_basename=model_save_basename
        )

        # 4. Loss scheduling to prevent too many losses optimized at the beginning at the same time
        li_scheduler = LossScheduler(args.alpha,
                                     int(args.warmup * 0.4),
                                     int(args.warmup),
                                     'linear')
        ld_scheduler = LossScheduler(args.beta,
                                     int(args.warmup * args.mask_warmup),
                                     int(args.warmup),
                                     'linear')
        lf_scheduler = LossScheduler(args.gamma,
                                     int(args.warmup * 0.2),
                                     int(args.warmup * 0.6),
                                     'linear')
        lambda_scheduler = LossScheduler(args.lambda_dd,
                                         int(args.warmup * 0.6),
                                         int(args.warmup),
                                         'sigmoid')

        print(f"Starting Stage 2 (FairGNN-WOD) training on {args.dataset}...")
        t_total = time.time()

        for epoch in range(args.epochs):
            t = time.time()

            alpha = li_scheduler.get_weight(epoch)
            beta = ld_scheduler.get_weight(epoch)
            gamma = lf_scheduler.get_weight(epoch)
            lambda_dd = lambda_scheduler.get_weight(epoch)

            mask = epoch > args.mask_warmup * args.warmup

            loss, lp, ld, li, lf, adv_loss = train_step(
                model, optimizer, adj, x, y, edge_index, mask,
                idx_train, args.adv_steps_per_epoch, alpha, beta, gamma, lambda_dd
            )

            # Validate
            val_metrics = evaluate(model, adj, x, y, edge_index, idx_val, s_true)

            if args.print_every != -1 and (epoch + 1) % args.print_every == 0:
                print(f"Epoch {epoch + 1} | Train Loss: {loss:.4f} | Adv Loss: {adv_loss:.4f} | "
                      f"LP: {lp:.4f} | LD: {ld:.4f} | LI: {li:.4f} | LF: {lf:.4f} | "
                      f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['acc']:.4f} | Time: {time.time() - t:.2f}s")

            if args.save_every != -1 and (epoch + 1) % args.save_every == 0:
                torch.save(model.state_dict(), f"{model_save_basename}_{epoch + 1}.pt")

            # Early stopping based on Val F1
            if early_stop.step(val_metrics['f1'], model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"FairGNN-WOD Training Finished. Total Time: {time.time() - t_total:.2f}s")

    # 4. Testing with the best model
    print("\nRestoring best model for final evaluation...")
    model.load_state_dict(torch.load(f"{model_save_basename}_best.pt", map_location=device))

    test_metrics = evaluate(model, adj, x, y, edge_index, idx_test, s_true=s_true)

    print("--------------------------------------------------")
    print(f"Test Set Results ({args.dataset}):")
    print(f"     AUC: {test_metrics['auc']:.4f}")
    print(f"Accuracy: {test_metrics['acc']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")

    if s_true is not None:
        print("Fairness Metrics:")
        for k, v in test_metrics.items():
            if k not in ['loss', 'acc', 'f1', 'auc']:
                print(f"  {k}: {format_metric(v, precision=4)}")
    print("--------------------------------------------------")
