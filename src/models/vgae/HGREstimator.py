#!/usr/bin/env/python

import torch
import torch.nn as nn


class HGREstimator(nn.Module):
    """
    Neural estimator for the Hirschfeld-Gebelein-Rényi (HGR) coefficient.

    The HGR coefficient measures the maximal correlation between two random variables $Z$ and $Y$.
    This module approximates HGR by learning two non-linear transformations, $f(Z)$ and $g(Y)$, that maximize
    $\mathbb{E}[f(Z)g(Y)]$ subject to $\mathbb{E}[f(Z)] = \mathbb{E}[g(Y)] = 0$
    and $\mathbb{E}[f^2(Z)] = \mathbb{E}[g^2(Y)] = 1$.

    In this pipeline, it serves as an upper bound for dependency during
    VGAE training to ensure the latent space $Z$ remains uncorrelated with
    the target labels $Y$, preventing label leakage into the embeddings which produce $S$.

    Parameters
    ----------
    z_dim : int
        Dimension of the latent representation $Z$.
    y_dim : int
        Dimension of the label/target vector $Y$.
    hidden_dim : int, optional
        Width of the MLP hidden layers.
        Defaults to 32.
    """
    def __init__(self, z_dim: int, y_dim: int, hidden_dim: int = 32):
        super(HGREstimator, self).__init__()
        # We regress p(z) and p(y)
        # MLP for transforming Z
        self.f = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # MLP for transforming Y
        self.g = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the HGR estimate between two tensors.

        The forward pass normalizes the outputs of $f(Z)$ and $g(Y)$ to have
        zero mean and unit variance, making the expected product equivalent
        to the correlation coefficient.

        Parameters
        ----------
        z : torch.Tensor
            Latent embeddings of shape [N, z_dim].
        y : torch.Tensor
            Target labels or predictions of shape [N, y_dim].

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the estimated HGR correlation value in the range [0, 1].
        """
        # Estimate p(y) and p(z)
        p_z = self.f(z)
        p_y = self.g(y)
        # Batch-wise normalization to satisfy E[f] = 0 and E[f^2] = 1
        p_z = (p_z - p_z.mean()) / (p_z.std() + 1e-8)
        p_y = (p_y - p_y.mean()) / (p_y.std() + 1e-8)
        # Calculate HGR as E(p(y)p(z)) = mean of the product normalized inputs,
        # as E(p2(y)) = E(p2(z) = 1 and thus the denominator sqrt(E(p2(y)) * E(p2(z)) = 1
        hgr = torch.mean(p_z * p_y)

        return hgr
