#!/usr/bin/env/python
# Implementation of the Adaptive Assigner of FairGNN-WOD implemented as a MLP

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAssigner(nn.Module):
    def __init__(self, in_dim, num_channels, hidden_dim=64):
        super(AdaptiveAssigner, self).__init__()
        self.num_channels = num_channels

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim, bias=False)

        self.linear_out = nn.Linear(hidden_dim, num_channels)

    def forward(self, x, edge_index):
        # Get input and output node features
        vj, vi = edge_index
        xi = torch.index_select(x, 0, vi)
        xj = torch.index_select(x, 0, vj)

        # Concatenate features of each i,j pair, two weight and adding is the same as one weight 2*D, num_channels
        # but limits large matrices
        concatenated_features = F.relu(self.linear1(xi) + self.linear2(xj))

        # Get importance scores of v_j for each channel of v_i
        psi = self.linear_out(concatenated_features)

        # Get normalized weights omega_vi_vj by softmax(psi_vi_vj)
        omega = F.softmax(psi, dim=-1)

        return omega    # [E, C]
