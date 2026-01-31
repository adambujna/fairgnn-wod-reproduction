#!/usr/bin/env python

import torch
import math
import warnings


def format_metric(x, precision=4):
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f"{v:.{precision}f}" for v in x) + "]"
    else:
        return f"{x:.{precision}f}"


def remove_self_loops(edge_index):
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def negative_sampling(edge_index, num_nodes, num_neg, max_tries=3):
    pos = (edge_index[0] * num_nodes + edge_index[1])
    pos = torch.unique(pos)

    chance = 1.0 - (edge_index.size(1) / (num_nodes ** 2))
    sample_size = int(1.1 * num_neg / chance)

    neg_samples = []
    for _ in range(max_tries):
        neg = torch.randint(num_nodes * num_nodes, size=(sample_size,), device=pos.device)
        neg = neg[~torch.isin(neg, pos)]
        neg_samples.append(neg)

        neg_all = torch.unique(torch.cat(neg_samples))

        if neg_all.numel() >= num_neg:
            neg_all = neg_all[:num_neg]
            break
    else:
        neg_all = torch.unique(torch.cat(neg_samples))
        warnings.warn(f"Only {neg_all.numel()} negative samples found "
                      f"after f{max_tries} tries (requested {num_neg}). "
                      f"Increase the number of tries if necessary.", Warning)

    return torch.stack([neg_all // num_nodes, neg_all % num_nodes])


def rbf_kernel(x, y, sigma):
    XX = (x ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    YY = (y ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    dist = XX + YY.T - 2 * x @ y.T  # [B, B]
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd2_rbf(x, y, sigma):
    k_xx = rbf_kernel(x, x, sigma)
    k_yy = rbf_kernel(y, y, sigma)
    k_xy = rbf_kernel(x, y, sigma)
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def median_heuristic(h_sub, max_samples=1000):
    B, C, H = h_sub.shape
    X = h_sub.reshape(B * C, H)

    max_samples = min(max_samples, X.size(0))
    idx = torch.randperm(X.size(0))[:max_samples]

    X = X[idx]
    with torch.no_grad():
        XX = (X ** 2).sum(dim=1, keepdim=True)
        dist = XX + XX.T - 2 * X @ X.T
        dist = dist[dist > 0]
        sigma = torch.sqrt(0.5 * dist.median())

    return sigma


def get_baseline_entropy(s_true):
    p1 = s_true.float().mean().item()
    p0 = 1.0 - p1

    ent = 0
    if p0 > 0:
        ent -= p0 * math.log(p0)
    if p1 > 0:
        ent -= p1 * math.log(p1)

    return ent
