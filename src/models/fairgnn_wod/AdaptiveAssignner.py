#!/usr/bin/env/python

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAssigner(nn.Module):
    """
    The "adaptive assigner" mechanism for channel-specific neighborhood aggregation.

    Unlike standard Graph Attention Networks (GAT) that produce a single scalar
    weight per edge, the `AdaptiveAssigner` produces a vector of weights $\Omega_{ij}$
    of size $C$ (number of channels).
    This allows the model to selectively aggregate features from neighbors differently for each latent disentangled
    component.

    Parameters
    ----------
    in_dim : int
        Dimension of the input node features.
    num_channels : int
        The number of disentangled latent channels ($C$).
    hidden_dim : int, optional
        Dimension of the internal hidden representation.
        Defaults to 64.
    """
    def __init__(self, in_dim: int, num_channels: int, hidden_dim: int = 64):
        super(AdaptiveAssigner, self).__init__()
        self.num_channels = num_channels

        # Separate linear layers for source and target nodes
        # as optimization to prevent large [2*D, H] weight matrices if concatenating.
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim, bias=False)

        self.linear_out = nn.Linear(hidden_dim, num_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Computes channel-wise assignment weights for every edge in the graph.

        The operation follows:
        $\psi_{ij} = \text{Linear}(\text{ReLU}(W_1 x_i + W_2 x_j))$
        $\omega_{ij} = \text{Softmax}(\psi_{ij})$

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [N, in_dim].
        edge_index : torch.Tensor
            Graph connectivity in COO format [2, E].

        Returns
        -------
        torch.Tensor
            Normalized assignment weights $\omega$ of shape [E, num_channels],
            where $\sum_{c=1}^C \omega_{ij, c} = 1.0$ for each edge.
        """
        # Get input (i) and output (j) node features
        vj, vi = edge_index
        xi = torch.index_select(x, 0, vi)
        xj = torch.index_select(x, 0, vj)

        # Compute pair-wise relational features
        # xi + xj via separate linear layers is mathematically equivalent to
        # [xi, xj] @ W but more memory efficient.
        concatenated_features = F.relu(self.linear1(xi) + self.linear2(xj))

        # Get importance scores of v_j for each channel of v_i
        psi = self.linear_out(concatenated_features)

        # Get normalized weights across channels
        omega = F.softmax(psi, dim=-1)

        return omega    # [E, C]
