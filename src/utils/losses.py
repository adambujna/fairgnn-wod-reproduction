#!/usr/bin/env python

import torch
import torch.nn.functional as F
from src.utils.utils import median_heuristic, mmd2_rbf


def structure_reconstruction_loss(pos_edge_logits, neg_edge_logits):
    loss_pos = 0.0
    if pos_edge_logits is not None:
        loss_pos = F.binary_cross_entropy_with_logits(pos_edge_logits, torch.ones_like(pos_edge_logits))

    loss_neg = 0.0
    if neg_edge_logits is not None:
        loss_neg = F.binary_cross_entropy_with_logits(neg_edge_logits, torch.zeros_like(neg_edge_logits))

    return loss_pos + loss_neg


def kl_divergence(mean, log_var):
    return -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))


def negative_entropy(probabilities):
    return torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))


def prediction_loss(y_logits, y_true):
    return F.cross_entropy(y_logits, y_true)


def demographic_loss(logits_c, s_true, smoothing=0.1):
    loss = 0.0
    C = logits_c.size(0)
    for c in range(C):
        loss += F.cross_entropy(logits_c[c], s_true, label_smoothing=smoothing)
    return loss / C


def fairness_loss(y_logits, s_true):
    y = y_logits[:, 1]
    s = s_true.float()

    y = y - y.mean()
    s = s - s.mean()

    cov = torch.mean(s * y)
    std = y.std() * s.std() + 1e-8
    return torch.abs(cov / std)


def independence_loss(h, kernel="rbf", batch_size=256, sigma=1.0):    # [N, C, H]
    """Not for use with the full graph with an RBF kernel."""
    if kernel == "linear":
        # If we do a linear kernel, then MMD^2(k1,k2) simplifies to (mean_n(k1) * mean_n(k2))^2
        # and the sum_{i<j} of (mean_n(k1) * mean_n(k2))^2 is further equivalent to the equation below
        mu = h.mean(axis=0)     # [C, H]
        C = mu.size(0)
        loss = C * (mu.pow(2).sum()) - mu.sum(dim=0).pow(2).sum()
    else:   # rbf
        N, C, H = h.shape
        B = min(N, batch_size)
        idx = torch.randperm(N)[:B]
        h_sub = h[idx]  # [B, C, H]
        h_sub = F.normalize(h_sub, dim=2)

        if sigma is None:
            sigma = median_heuristic(h_sub)

        loss = 0.0
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                X = h_sub[:, c1, :]
                Y = h_sub[:, c2, :]
                loss += mmd2_rbf(X, Y, sigma)

        return -loss / (C * (C - 1))
