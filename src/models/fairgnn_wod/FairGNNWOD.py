#!/usr/bin/env/python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.utils.utils import get_baseline_entropy
from src.models.fairgnn_wod.AdaptiveAssignner import AdaptiveAssigner
from src.models.fairgnn_wod.DemographicDiscriminator import DemographicClassifier, MaskMechanism
from src.models.fairgnn_wod.DisentangledGraphConvEncoder import DisentangledGraphConvEncoder


class FairGNNWOD(nn.Module):
    def __init__(self, vgae, input_size, hidden_size, num_channels):
        super(FairGNNWOD, self).__init__()

        # --- Stage 1 ---
        self.vgae = vgae
        self.vgae.eval()
        for p in self.vgae.parameters():
            p.requires_grad = False

        # --- Stage 2 ---
        self.adaptive_assigner = AdaptiveAssigner(input_size, num_channels)
        self.disentangled_graph_encoder = DisentangledGraphConvEncoder(input_size, hidden_size, num_channels)
        self.demographic_classifier = DemographicClassifier(hidden_size)
        self.mask_mechanism = MaskMechanism(hidden_size, num_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_ * grad_output, None

    def grad_reverse(self, x, lambda_=1.0):
        return self.GradReverse.apply(x, lambda_)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if module is not self.vgae:
                module.train(mode)
        return self

    def forward(self, adj, x, edge_index, lambda_, mask):
        # --- Stage 1 ---
        # Infer missing demographics
        self.vgae.eval()
        with torch.no_grad():
            _, _, _, s_logits, _, _, _ = self.vgae(adj, x)
        s_pred = s_logits.detach().argmax(dim=-1)   # [N]
        H_s = get_baseline_entropy(s_pred)

    # --- Stage 2 ---
        # 1. Get omega (which neighbor is relevant to each channel)
        omega = self.adaptive_assigner(x, edge_index)   # [E, C]

        # 2. Get disentangled channel representations [N, C, H] and node features by channel [C, N * H]
        h = self.disentangled_graph_encoder(x, edge_index, omega)   # [N, C, H]
        N, C, H = h.shape

        # 3. Identify channels containing demographic information
        # h shape: [N, C, H]
        h_adv = self.grad_reverse(h, lambda_=lambda_)

        # This treats every channel of every node as a separate "sample"
        h_flat = h_adv.reshape(N * C, H)  # [N*C, H]
        all_logits = self.demographic_classifier(h_flat)  # [N*C, 2]
        c_logits = all_logits.reshape(N, C, 2).transpose(0, 1)  # [C, N, 2]
        # Get channel sensitivity
        s_expanded = s_pred.repeat(C)
        all_losses = F.cross_entropy(c_logits.reshape(C * N, 2), s_expanded.long(), reduction='none')
        channel_losses = all_losses.reshape(C, N).mean(dim=1)  # Shape: [C]
        channel_sensitivity = (1.0 - (channel_losses / (H_s + 1e-8))).clamp(0.0, 1.0)

        # 4. Mask h
        h_masked = h
        if mask:
            h_masked = self.mask_mechanism(h, channel_sensitivity)  # [N, C, H]

        # 5. Label prediction
        h_flat = h_masked.reshape((N, C * H))
        y_logits = self.classifier(h_flat)

        return y_logits, h, c_logits, s_pred
