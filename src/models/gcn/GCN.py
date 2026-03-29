#!/usr/bin/env/python
# Implementation in PyTorch matching variant used by FairGNN:
# https://github.com/EnyanDai/FairGNN/blob/main/src/models/GCN.py.

import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """
    A single Graph Convolutional layer.

    This implementation performs the message passing operation:
    $Z = \sigma(\hat{A}XW + b)$, where $\hat{A}$ is the normalized
    adjacency matrix provided by the `GraphDataset`.

    Parameters
    ----------
    in_features : int
        Dimension of input node features.
    out_features : int
        Dimension of output node representations.
    weight : bool, optional
        Whether to include a learnable weight matrix $W$.
        Defaults to True.
    bias : bool, optional
        Whether to include a learnable bias vector $b$.
        Defaults to True.
    activation : callable, optional
        An activation function (e.g., `F.relu`) applied after aggregation.
        Defaults to None.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 weight: bool = True,
                 bias: bool = True,
                 activation: object = None):
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

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the graph convolution operation.

        Parameters
        ----------
        adj : torch.Tensor (Sparse or Dense)
            The (normalized) adjacency matrix of shape [N, N].
        x : torch.Tensor
            Node feature matrix of shape [N, in_features].

        Returns
        -------
        torch.Tensor
            Transformed node representations of shape [N, out_features].
        """
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
    """
    A 2-layer GCN backbone for extracting graph-structured features.

    This module acts as the 'encoder' part of the GCN, producing latent
    representations that can be reused by different heads.

    Parameters
    ----------
    input_size : int
        Number of input features per node.
    hidden_size : int
        Dimension of the hidden layers.
    p_dropout : float, optional
        Dropout probability applied after the first layer.
        Defaults to 0.0.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 p_dropout: float = 0.0):
        super(BackboneGCN, self).__init__()

        self.gc1 = GraphConv(input_size, hidden_size, True, True, None)
        self.gc2 = GraphConv(hidden_size, hidden_size, True, True, None)
        self.dropout = nn.Dropout(p_dropout)
        self.ReLU = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent representation for all nodes.

        Parameters
        ----------
        g : torch.Tensor
            Adjacency matrix [N, N].
        x : torch.Tensor
            Input node features [N, input_size].

        Returns
        -------
        torch.Tensor
            Latent node embeddings [N, hidden_size].
        """
        x = self.ReLU(self.gc1(g, x))
        x = self.dropout(x)
        x = self.gc2(g, x)

        return x


class GCN(nn.Module):
    """
    A complete Graph Convolutional Network for node classification.

    This model wraps a `BackboneGCN` and adds a final linear classification layer.

    Parameters
    ----------
    input_size : int
        Number of input features per node.
    hidden_size : int
        Number of hidden units in the backbone.
    output_size : int
        Number of output classes.
    p_dropout : float, optional
        Dropout probability.
        Defaults to 0.0.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 p_dropout: float = 0.0):
        super(GCN, self).__init__()

        self.backbone = BackboneGCN(input_size, hidden_size, p_dropout)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass for classification.

        Parameters
        ----------
        g : torch.Tensor
            Adjacency matrix [N, N].
        x : torch.Tensor
            Input node features [N, input_size].

        Returns
        -------
        torch.Tensor
            Classification logits of shape [N, num_classes].
        """
        x = self.backbone(g, x)
        x = self.fc1(x)
        return x
