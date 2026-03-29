#!/usr/bin/env/python

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TypeAlias
from torch_geometric.nn import MessagePassing

# We will use the GCN as the encoder of the VAE.
from src.models.gcn.GCN import BackboneGCN


VGAEOutput: TypeAlias = tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class PredictedAdjConv(MessagePassing):
    """
    A message-passing layer that uses predicted edge weights for aggregation.

    Unlike standard GCN layers that use a fixed adjacency matrix, this layer
    weights neighborhood messages by the logits/probabilities generated
    by the VGAE's structural decoder.

    Parameters
    ----------
    in_features : int
        Dimension of the input node features (typically the inferred $S$ or its latent).
    out_features : int
        Dimension of the output features after aggregation.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor, a_pos_logits: torch.Tensor) -> torch.Tensor:
        """
        Runs message passing where $x_{new} = \sum_{j \in \mathcal{N}(i)} \hat{A}_{ij} \cdot Wx_j$.

        Parameters
        ----------
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges].
        x : torch.Tensor
            Node features to be propagated.
        a_pos_logits : torch.Tensor
            Predicted edge weights/logits for the corresponding `edge_index`.

        Returns
        -------
        torch.Tensor
            Aggregated node representations.
        """
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_weight=a_pos_logits)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Defines how a message is propagated: Wx_j."""
        return edge_weight.unsqueeze(-1) * x_j


class DemographicVGAE(nn.Module):
    """
    Variational Graph Auto-Encoder for Demographic Disentanglement.

    This model implements the causal VGAE pipeline: $X, A \rightarrow Z \rightarrow S \rightarrow A', X'$.
    Its goal is to learn a latent space $Z$ that can infer sensitive attributes $S$,
    while recovering the original graph $(A, X)$ from those demographic attributes alone.

    Parameters
    ----------
    input_size : int
        Number of original node features.
    hidden_size : int
        Dimension of the GCN backbone and internal layers.
    latent_size : int
        Dimension of the stochastic latent space $Z$.
    """
    def __init__(self, input_size: int, hidden_size: int, latent_size: int):
        super(DemographicVGAE, self).__init__()

        # Encoder - Q(Z | A, X)
        self.encoder = BackboneGCN(input_size, hidden_size)
        # Variational heads
        self.mean_head = nn.Linear(hidden_size, latent_size)
        self.logvar_head = nn.Linear(hidden_size, latent_size)

        # Demographic Inference - Q(S | Z)
        self.s_decoder = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 2)
        )

        # P(A|S)
        self.structure_representation_dim = 32
        self.a_decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.structure_representation_dim)
        )

        # P(X|S,A)
        self.x_decoder_gcn = PredictedAdjConv(2, hidden_size)
        self.x_decoder_fc = nn.Linear(hidden_size, input_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick: $z = \mu + \epsilon \odot \sigma$.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, g: torch.Tensor, x: torch.Tensor):
        """
        Maps input graph to the latent distribution parameters.

        Returns
        -------
        mu : torch.Tensor
            Mean of the latent Gaussian.
        logvar : torch.Tensor
            Log-variance of the latent Gaussian.
        z : torch.Tensor
            Sampled latent vector.
        """
        h = self.encoder(g, x)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return mu, logvar, z

    def forward(self,
                g: torch.Tensor,
                x: torch.Tensor,
                pos_edge_index: torch.Tensor = None,
                neg_edge_index: torch.Tensor = None) -> VGAEOutput:
        """
        Full generative forward pass.

        1. **Encoder**: $Q(Z|X,A)$ produces latent $Z$.
        2. **S-Inference**: $Q(S|Z)$ predicts sensitive attributes.
        3. **A-Decoder**: $P(A|S)$ reconstructs structure via inner product of $S$-embeddings.
        4. **X-Decoder**: $P(X|A,S)$ reconstructs features using predicted structure.

        Parameters
        ----------
        g : torch.Tensor
            Input adjacency matrix [N, N].
        x : torch.Tensor
            Input node features [N, D].
        pos_edge_index : torch.Tensor, optional
            Positive edges for structural reconstruction.
        neg_edge_index : torch.Tensor, optional
            Negative edges for structural reconstruction.

        Returns
        -------
        tuple
            (x_hat, pos_a_logits, neg_a_logits, s_logits, mu, logvar, z)
            - x_hat: Reconstructed node features.
            - pos_a_logits/neg_a_logits: Predicted edge existence probabilities for A reconstruction loss.
            - s_logits: Inferred sensitive attributes.
            - mu, logvar, z: Latent space parameters and sample.
        """
        # 1. Encode Z: Q(Z | A, X)
        h = self.encoder(g, x)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        # 2. Infer S: Q(S | Z)
        s_logits = self.s_decoder(z)
        s_sample = F.softmax(s_logits, dim=-1)

        # Infer structure: P(A | S)
        s_struct = self.a_decoder(s_sample)
        pos_a_logits = None
        neg_a_logits = None
        if pos_edge_index is not None:
            pos_a_logits = (s_struct[pos_edge_index[0]] * s_struct[pos_edge_index[1]]).sum(dim=-1)
        if neg_edge_index is not None:
            neg_a_logits = (s_struct[neg_edge_index[0]] * s_struct[neg_edge_index[1]]).sum(dim=-1)

        # Decode features: P(X | A, S)
        x_hat = None
        if pos_edge_index is not None:
            x_hat = self.x_decoder_gcn(pos_edge_index, s_sample, pos_a_logits)
            x_hat = self.x_decoder_fc(x_hat)

        return x_hat, pos_a_logits, neg_a_logits, s_logits, mu, logvar, z
