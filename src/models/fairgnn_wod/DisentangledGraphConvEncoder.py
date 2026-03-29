#!/usr/bin/env python

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class DisentangledGraphConv(MessagePassing):
    """
    A Disentangled Graph Convolutional layer.

    Unlike a standard GCN that aggregates all features uniformly, this layer maintains $C$ separate latent channels.
    It performs independent message passing for each channel $c$ using the channel-specific attention weights
    $\omega_c$, followed by a channel-specific linear transformation.

    The forward pass operation per channel is roughly:
    $\mathbf{h}_{i, c}^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \omega_{ij, c} \mathbf{h}_{j, c}^{(l)} \mathbf{W}_c^{(l)} + \mathbf{b}_c \right)$

    Parameters
    ----------
    in_size : int
        Dimension of the input features ($D$ for the first layer, $H$ otherwise).
    out_size : int
        Dimension of the output features per channel ($H$).
    num_channels : int
        Number of disentangled channels ($C$).
    weight : bool, optional
        If True, applies a channel-specific weight matrix.
        Defaults to True.
    bias : bool, optional
        If True, applies a channel-specific bias vector.
        Defaults to False.
    activation : callable, optional
        Activation function to apply after the transformation.
        Defaults to None.
    """
    def __init__(self, in_size: int, out_size: int, num_channels: int,
                 weight: bool = True, bias: bool = False, activation: object = None):
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
        """Standard initialization of weight and bias."""
        nn.init.xavier_uniform_(self.proj)
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Constructs messages for a specific channel.

        Parameters
        ----------
        x_j : torch.Tensor
            Features of the source nodes for the current channel, shape [E, H].
        edge_weight : torch.Tensor
            The adaptive assignment weights $\omega_c$ for the current channel, shape [E].

        Returns
        -------
        torch.Tensor
            Weighted messages of shape [E, H].
        """
        return edge_weight.view(-1, 1) * x_j

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Executes the channel-wise graph convolution.

        If the input is 2D (initial layer), it uses `self.proj` to map the
        input features into the 3D channel space $[N, C, H]$.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
            Shape [N, D] for the first layer, or [N, C, in_size] for subsequent layers.
        edge_index : torch.Tensor
            Graph connectivity, shape [2, E].
        omega : torch.Tensor
            Adaptive assignment weights for all channels, shape [E, C].

        Returns
        -------
        torch.Tensor
            Updated disentangled embeddings of shape [N, C, out_size].
        """
        num_nodes = x.size(0)

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
    """
    A multi-layer encoder for disentangled graph representations.

    This module stacks multiple `DisentangledGraphConv` layers.
    The first layer handles the projection from standard node features into the
    channelized latent space $[N, C, H]$, while subsequent layers transform those channel embeddings.

    Parameters
    ----------
    in_size : int
        Dimension of the raw input node features.
    out_size : int
        Hidden dimension size per channel ($H$).
    num_channels : int
        Number of disentangled channels ($C$).
    num_layers : int, optional
        Total number of convolutional layers.
        Defaults to 2.
    """
    def __init__(self, in_size: int, out_size: int, num_channels: int, num_layers: int = 2):
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the stacked disentangled convolutions.

        Applies Layer Normalization and ReLU activations between the
        convolutional steps to stabilize the channel representations.

        Parameters
        ----------
        x : torch.Tensor
            Raw input node features, shape [N, in_size].
        edge_index : torch.Tensor
            Graph connectivity [2, E].
        omega : torch.Tensor
            Adaptive assignment weights, shape [E, C].

        Returns
        -------
        torch.Tensor
            Final disentangled latent representations, shape [N, C, H].
        """
        h = x

        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, omega)
            h = self.layer_norm(h)
            if i < len(self.layers) - 1:
                h = self.ReLU(h)

        return h    # [N, C, H]
