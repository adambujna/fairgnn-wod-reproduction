#!/usr/bin/env python

import torch
import torch.nn.functional as F
from src.utils.utils import median_heuristic, mmd2_rbf


def structure_reconstruction_loss(pos_edge_logits: torch.Tensor, neg_edge_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the binary cross-entropy loss for graph structure reconstruction.

    Used in the VGAE (Stage 1) to ensure the latent embeddings preserve the
    topological properties of the original graph.

    Parameters
    ----------
    pos_edge_logits : torch.Tensor
        Logits for existing edges (positive samples).
    neg_edge_logits : torch.Tensor
        Logits for non-existent edges (negative samples).

    Returns
    -------
    torch.Tensor
        The sum of positive and negative reconstruction losses.
    """
    loss_pos = 0.0
    if pos_edge_logits is not None:
        loss_pos = F.binary_cross_entropy_with_logits(pos_edge_logits, torch.ones_like(pos_edge_logits))

    loss_neg = 0.0
    if neg_edge_logits is not None:
        loss_neg = F.binary_cross_entropy_with_logits(neg_edge_logits, torch.zeros_like(neg_edge_logits))

    return loss_pos + loss_neg


def kl_divergence(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence for a Gaussian distribution.

    Measures the distance between the learned latent distribution $\mathcal{Q}(Z|X,A)$
    and the prior $\mathcal{P}(Z) = \mathcal{N}(0, I)$.

    Parameters
    ----------
    mean : torch.Tensor
        The mean ($\mu$) of the latent distribution.
    log_var : torch.Tensor
        The log-variance ($\log \sigma^2$) of the latent distribution.

    Returns
    -------
    torch.Tensor
        The mean KL divergence across the batch/nodes.
    """
    return -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))


def negative_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """
    Calculates the negative Shannon entropy of a probability distribution.

    Can be used to enforce higher or lower entropy for S predictions.

    Parameters
    ----------
    probabilities : torch.Tensor
        Predicted probabilities, shape [N, num_classes].

    Returns
    -------
    torch.Tensor
        The mean negative entropy: $\frac{1}{N} \sum p \log(p)$.
    """
    return torch.mean(torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1))


def prediction_loss(y_logits, y_true):
    """
    Calculates the prediction loss $L_P$ from the main loss of fairGNN-WOD.

    This is equivalent to the cross-entropy loss.

    Parameters
    ----------
    y_logits : torch.Tensor
        Model output logits, shape [N, 2].
    y_true : torch.Tensor
        True labels.

    Returns
    -------
    torch.Tensor
        The cross-entropy loss.
    """
    return F.cross_entropy(y_logits, y_true)


def demographic_loss(logits_c: torch.Tensor, s_true: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """
    Computes the cross-entropy loss for predicting sensitive attributes.
    
    This is $L_D$ from the paper and forms the discriminator loss.

    Iterates through each latent channel to measure how much demographic information ($s$) is leaked by that channel.
    Used to calculate channel sensitivity.

    Parameters
    ----------
    logits_c : torch.Tensor
        Logits for each channel, shape [num_channels, N, num_sensitive_classes].
    s_true : torch.Tensor
        Ground truth sensitive attributes.
    smoothing : float, optional
        Label smoothing factor for cross-entropy. Defaults to 0.1.

    Returns
    -------
    torch.Tensor
        The average cross-entropy loss across all channels.
    """
    loss = 0.0
    C = logits_c.size(0)
    for c in range(C):
        loss += F.cross_entropy(logits_c[c], s_true, label_smoothing=smoothing)
    return loss / C


def fairness_loss(y_logits: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the absolute Pearson Correlation between predictions and sensitive attributes.

    This is $L_F$ from the paper.

    Acts as a proxy for Statistical Parity.
    Minimizing this covariance encourages the model to make predictions independently of the sensitive group.

    Parameters
    ----------
    y_logits : torch.Tensor
        Model output logits, shape [N, 2].
    s_true : torch.Tensor
        Binary sensitive attributes.

    Returns
    -------
    torch.Tensor
        The absolute correlation coefficient $|\rho_{y,s}|$.
    """
    y = y_logits[:, 1]
    s = s_true.float()

    y = y - y.mean()
    s = s - s.mean()

    cov = torch.mean(s * y)
    std = y.std() * s.std() + 1e-8
    return torch.abs(cov / std)


def independence_loss(h: torch.Tensor,
                      kernel: str = "rbf",
                      batch_size: int = 256,
                      sigma: float = 1.0) -> torch.Tensor:
    """
    Computes a loss to encourage independence between latent channels.

    This is $L_I$ from the paper.

    Note: The RBF implementation uses a sampled batch for efficiency.

    Parameters
    ----------
    h : torch.Tensor
        Latent representations of shape [N, C, H], where C is the number of
        channels and H is the feature dimension.
    kernel : str, optional
        Type of kernel to use ('linear' or 'rbf').
        Defaults to "rbf".
    batch_size : int, optional
        Size of the random subset used for RBF calculation.
        Defaults to 256.
    sigma : float, optional
        RBF bandwidth.
        If None, it is estimated via `src.utils.median_heuristic`.

    Returns
    -------
    torch.Tensor
        The negated average independence score.
    """
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

    # Return negated average to be used as a minimization objective
    return -loss / (C * (C - 1))
