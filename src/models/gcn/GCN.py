#!/usr/bin/env/python
# Implementation in PyTorch matching variant used by FairGNN:
# https://github.com/EnyanDai/FairGNN/blob/main/src/models/GCN.py.

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """Graph Convolution."""
    def __init__(self, in_features, out_features, weight=True, bias=True, activation=None):
        super(GraphConv, self).__init__()
        self._activation = activation

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        # 1. H = XW
        if self.weight is not None:
            h = torch.matmul(x, self.weight)

        # 2. Neighborhood aggregation Z = AH
        if adj.is_sparse:
            z = torch.sparse.mm(adj, h)
        else:
            z = torch.matmul(adj, h)

        # 3. Bias Z = AH + B
        if self.bias is not None:
            z += self.bias

        # 4. Activation
        if self._activation is not None:
            return self._activation(z)

        return z


class BackboneGCN(nn.Module):
    """Graph Convolutional Network without classification layer."""
    def __init__(self, input_size, hidden_size, p_dropout=0.0):
        super(BackboneGCN, self).__init__()

        self.gc1 = GraphConv(input_size, hidden_size, True, True, None)
        self.gc2 = GraphConv(hidden_size, hidden_size, True, True, None)
        self.dropout = nn.Dropout(p_dropout)
        self.ReLU = nn.ReLU()

    def forward(self, g, x):
        x = self.ReLU(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)

        return x


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, p_dropout=0.0):
        super(GCN, self).__init__()

        self.backbone = BackboneGCN(input_size, hidden_size, p_dropout)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, g, x):
        x = self.backbone(g, x)
        x = self.fc1(x)
        return x
