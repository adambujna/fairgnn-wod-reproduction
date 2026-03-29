#!/usr/bin/env python

import torch
import math
import warnings


def format_metric(x: float | list[float] | tuple[float], precision: int = 4) -> str:
    """
    Formats a single numeric metric or a per-class fairness metric into a rounded string.

    This is particularly useful for logging results,
    especially those that contain per-class values like fairness metrics.

    Parameters
    ----------
    x : float or list[float] or tuple[float]
        The metric value(s) to format.
        Can be a single scalar or a collection (list/tuple) of floats.
    precision : int, optional
        The number of decimal places to include.
        Defaults to 4.

    Returns
    -------
    str
        A string representation of the input.
        Sequences are wrapped in brackets and separated by commas.
    """
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f"{v:.{precision}f}" for v in x) + "]"
    else:
        return f"{x:.{precision}f}"


def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Removes edges from an edge index where the source and target nodes are the same.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity in COO format, shape [2, num_edges].

    Returns
    -------
    torch.Tensor
        Edge index tensor with self-loops filtered out, shape [2, num_edges_new].
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def negative_sampling(edge_index: torch.Tensor,
                      num_nodes: int,
                      num_neg: int,
                      max_tries: int = 3) -> torch.Tensor:
    """
    Samples negative edges (non-existent links) from the graph.

    This function treats the adjacency matrix as a flattened 1D array of size
    $N^2$ to efficiently sample pairs that do not exist in the positive
    `edge_index`.

    Parameters
    ----------
    edge_index : torch.Tensor
        Positive edge connectivity in COO format [2, num_edges].
    num_nodes : int
        Total number of nodes in the graph ($N$).
    num_neg : int
        The desired number of negative edges to sample.
    max_tries : int, optional
        Number of sampling iterations before giving up.
        Defaults to 3.

    Returns
    -------
    torch.Tensor
        Negative edge index in COO format [2, num_neg].

    Warns
    -----
    Warning
        If the requested `num_neg` cannot be reached within `max_tries`.
    """
    # Map 2D indices (i, j) to 1D: index = i * N + j
    pos = (edge_index[0] * num_nodes + edge_index[1])
    pos = torch.unique(pos)

    # Estimate sample size based on graph density to minimize iterations
    chance = 1.0 - (edge_index.size(1) / (num_nodes ** 2))
    sample_size = int(1.1 * num_neg / chance)

    neg_samples = []
    for _ in range(max_tries):
        # Sample random indexes [0, N^2)
        neg = torch.randint(num_nodes * num_nodes, size=(sample_size,), device=pos.device)
        # Filter out indices that exist in the positive set
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

            # Map 1D indices back to 2D (row = idx // N, col = idx % N)
    return torch.stack([neg_all // num_nodes, neg_all % num_nodes])


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Computes the Gaussian RBF kernel matrix.

    The kernel is calculated as $K(x, y) = \exp(-\frac{\|x - y\|^2}{2\sigma^2})$.
    It uses the expansion $\|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2x^\top y$ for
    efficient computation.

    Parameters
    ----------
    x : torch.Tensor
        Input features of shape [N, D].
    y : torch.Tensor
        Input features of shape [M, D].
    sigma : float
        The bandwidth parameter controlling the kernel width.

    Returns
    -------
    torch.Tensor
        The pairwise kernel matrix, shape [N, M].
        Kernel matrix[i, j] is the distance between x[i] and y[j].
    """
    XX = (x ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    YY = (y ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    dist = XX + YY.T - 2 * x @ y.T  # [B, B]
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd2_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Calculates the squared Maximum Mean Discrepancy (MMD) using an RBF kernel.

    MMD measures the distance between two distributions $P$ and $Q$ by
    comparing their mean embeddings in a Reproducing Kernel Hilbert Space (RKHS).

    Parameters
    ----------
    x : torch.Tensor
        Samples from distribution $P$ with shape [N, D].
    y : torch.Tensor
        Samples from distribution $Q$ with shape [M, D].
    sigma : float
        Bandwidth for the underlying RBF kernel.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the squared MMD distance.
    """
    k_xx = rbf_kernel(x, x, sigma)
    k_yy = rbf_kernel(y, y, sigma)
    k_xy = rbf_kernel(x, y, sigma)
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def median_heuristic(h_sub: torch.Tensor, max_samples: int = 1000) -> torch.Tensor:
    """
    Estimates the optimal RBF bandwidth ($\sigma$) using the median heuristic.

    This heuristic sets $\sigma$ based on the median of pairwise distances between
    samples, ensuring the kernel is sensitive to the data's scale.

    Parameters
    ----------
    h_sub : torch.Tensor
        Input tensor of shape [Batch, Channels, Hidden_Dimension].
    max_samples : int, optional
        Maximum number of samples to use for the estimate to save memory.
        Defaults to 1000.

    Returns
    -------
    torch.Tensor
        The calculated bandwidth $\sigma$ as a scalar tensor.
    """
    B, C, H = h_sub.shape
    X = h_sub.reshape(B * C, H)

    max_samples = min(max_samples, X.size(0))
    idx = torch.randperm(X.size(0))[:max_samples]

    X = X[idx]
    with torch.no_grad():
        XX = (X ** 2).sum(dim=1, keepdim=True)
        dist = XX + XX.T - 2 * X @ X.T
        # Extract strictly positive distances to avoid log(0) or zero medians
        dist = dist[dist > 0]
        sigma = torch.sqrt(0.5 * dist.median())

    return sigma


def get_baseline_entropy(s_true: torch.Tensor) -> float:
    """
    Computes the Shannon entropy of the binary sensitive attribute distribution.

    In the fGNN-WOD pipeline, this serves as the theoretical maximum loss for a non-informative predictor.
    It is used to normalize cross-entropy losses of the discriminator,
    ensuring that leakage scores are relative to the inherent uncertainty (imbalance) of the demographic groups.

    Parameters
    ----------
    s_true : torch.Tensor
        1D tensor of binary labels/sensitive attributes.

    Returns
    -------
    float
        The calculated entropy value.
    """
    p1 = s_true.float().mean().item()
    p0 = 1.0 - p1

    ent = 0
    if p0 > 0:
        ent -= p0 * math.log(p0)
    if p1 > 0:
        ent -= p1 * math.log(p1)

    return ent
