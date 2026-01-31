#!/usr/bin/env/python
# Implementation of the VAE which reconstructs graph structure (A', X') and complete demographics (S')
# from a graph (X, A) that is (partially) missing demographics (S)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# We will use the GCN as the encoder of the VAE.
from src.models.gcn.GCN import BackboneGCN


class PredictedAdjConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, edge_index, x, a_pos_logits):
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_weight=a_pos_logits)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j


class DemographicVGAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
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

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, g, x):
        h = self.encoder(g, x)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return mu, logvar, z

    def forward(self, g, x, pos_edge_index=None, neg_edge_index=None):  # adj[N, N], [N, D]
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
