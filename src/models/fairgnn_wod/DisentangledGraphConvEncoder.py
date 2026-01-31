#!/usr/bin/env python
# Implementation of a Disentangled Graph Convolution and the multichannel Disentangled Graph Encoder

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class DisentangledGraphConv(MessagePassing):
    def __init__(self, in_size, out_size, num_channels, weight=True, bias=False, activation=None):
        """
        Disentangled graph convolution layer, which works exactly like a Graph Convolution
        but maintains N_c separate channels.
        """
        super(DisentangledGraphConv, self).__init__()
        self._activation = activation
        self.in_size = in_size
        self.num_channels = num_channels
        self.out_size = out_size

        # F_c^R channel-wise input projection
        self.proj = nn.Parameter(torch.Tensor(in_size, num_channels, out_size))

        # W_c^(l) channel-wise propagation weights for this convolutional layer
        if weight:
            self.weight = nn.Parameter(torch.Tensor(num_channels, out_size, out_size))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_channels, out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj)
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def message(self, x_j, edge_weight):
        # x_j: [E, H] features of source nodes
        # edge_weight: [E] per-edge weight
        return edge_weight.view(-1, 1) * x_j

    def forward(self, x, edge_index, omega):
        num_nodes = x.size(0)
        src, dst = edge_index

        # Initial channel-wise transform
        # The first layer of X.shape [N, D], needs to separate features into channels [N, C, H]
        if x.dim() == 2:    # multiply by linear projection of shape [D, C, H]
            h = x @ self.proj.reshape(self.in_size, self.num_channels * self.out_size)
            h = h.reshape((num_nodes, self.num_channels, self.out_size))
        else:
            h = x

        # Channel-wise message passing
        N, C, H = h.size()
        # omega.shape = [E, C]
        # weight.shape = [C, H, C, H]

        # Gather features of all source nodes
        h_next = torch.zeros_like(h, device=h.device)

        for c in range(C):
            edge_weight_c = omega[:, c]
            h_next[:, c, :] = self.propagate(edge_index, x=h[:, c, :], edge_weight=edge_weight_c, size=(N, N))

        # Apply weight and bias channel-wise like a regular graph convolution
        # Channel-wise weight
        if self.weight is not None:
            h_next_reshaped = h_next.permute(1, 0, 2)  # [C, N, H]
            h_next_out = torch.bmm(h_next_reshaped, self.weight)  # [C, N, H] x [C, H, H] = [C, N, H]
            h_next = h_next_out.permute(1, 0, 2)  # back to [N, C, H]
        # Channel-wise bias
        if self.bias is not None:
            h_next += self.bias.unsqueeze(0)
        # Activation
        if self._activation is not None:
            h_next = self._activation(h_next)

        return h_next


class DisentangledGraphConvEncoder(nn.Module):
    def __init__(self, in_size, out_size, num_channels, num_layers=2):
        super(DisentangledGraphConvEncoder, self).__init__()
        self.num_channels = num_channels

        # We use several identical DisentangledGraphConv layers
        self.layers = nn.ModuleList()
        # 1. layer goes from [n, d] -> [n, c, h]
        self.layers.append(
            DisentangledGraphConv(in_size, out_size, num_channels)
        )
        # Subsequent layers [n, c, h] -> [n, c, h]
        for _ in range(num_layers - 1):
            self.layers.append(
                DisentangledGraphConv(out_size, out_size, num_channels)
            )
        self.layer_norm = nn.LayerNorm(out_size)
        self.ReLU = nn.ReLU()

    def forward(self, x, edge_index, omega):
        h = x

        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, omega)
            h = self.layer_norm(h)
            if i < len(self.layers) - 1:
                h = self.ReLU(h)

        return h    # [N, C, H]
