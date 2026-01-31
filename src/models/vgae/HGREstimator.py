#!/usr/bin/env/python
# Implementation of the approximation networks of p(z) and p(y) for HGR calculation

import torch
import torch.nn as nn


class HGREstimator(nn.Module):
    def __init__(self, z_dim, y_dim, hidden_dim=32):
        super(HGREstimator, self).__init__()
        # We regress p(z) and p(y)
        self.f = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.g = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z, y):
        # Estimate p(y) and p(z)
        p_z = self.f(z)
        p_y = self.g(y)
        # Normalize as per the paper
        p_z = (p_z - p_z.mean()) / (p_z.std() + 1e-8)
        p_y = (p_y - p_y.mean()) / (p_y.std() + 1e-8)
        # Calculate HGR as E(p(y)p(z)) = mean of the product normalized inputs,
        # as E(p2(y)) = E(p2(z) = 1 and thus the denominator sqrt(E(p2(y)) * E(p2(z)) = 1
        hgr = torch.mean(p_z * p_y)

        return hgr
