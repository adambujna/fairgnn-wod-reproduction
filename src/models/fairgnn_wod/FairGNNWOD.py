#!/usr/bin/env/python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Any

from src.utils.utils import get_baseline_entropy
from src.models.fairgnn_wod.AdaptiveAssignner import AdaptiveAssigner
from src.models.fairgnn_wod.DemographicDiscriminator import DemographicClassifier, MaskMechanism
from src.models.fairgnn_wod.DisentangledGraphConvEncoder import DisentangledGraphConvEncoder


class FairGNNWOD(nn.Module):
    """
    Fair Graph Neural Network Without Demographics (fairGNN-WOD).

    This model implements the entire model with the stage-2 training logic.
    It uses a pre-trained and frozen Stage-1 VGAE to infer sensitive attributes for all nodes,
    then learns a disentangled representation where sensitive information is
    actively identified via the discriminator and suppressed via a learnable masking mechanism.

    Parameters
    ----------
    vgae : DemographicVGAE
        The pre-trained DemographicVGAE from Stage 1.
    input_size : int
        Dimension of the input node features.
    hidden_size : int
        Feature dimension per latent channel ($H$).
    num_channels : int
        Number of disentangled channels ($C$).
    """
    def __init__(self, vgae: nn.Module, input_size: int, hidden_size: int, num_channels: int):
        super(FairGNNWOD, self).__init__()

        # --- Stage 1 (Frozen) ---
        self.vgae = vgae
        self.vgae.eval()
        for p in self.vgae.parameters():
            p.requires_grad = False

        # --- Stage 2 (Trainable) ---
        self.adaptive_assigner = AdaptiveAssigner(input_size, num_channels)
        self.disentangled_graph_encoder = DisentangledGraphConvEncoder(input_size, hidden_size, num_channels)
        self.demographic_classifier = DemographicClassifier(hidden_size)
        self.mask_mechanism = MaskMechanism(hidden_size, num_channels)

        # --- Task-specific classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    class GradReverse(Function):
        """
        Gradient Reversal Layer (GRL).

        During the forward pass, it acts as an identity function.
        During the backward pass, it multiplies the discriminator gradient by $-\lambda$ for the encoder.
        This forces the encoder to strip information that the
        demographic classifier uses to succeed.
        """
        @staticmethod
        def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            return -ctx.lambda_ * grad_output, None

    def grad_reverse(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """Gradient Reversal."""
        return self.GradReverse.apply(x, lambda_)

    def train(self, mode: bool = True) -> nn.Module:
        """Sets the model into training mode. Skips the VGAE which is frozen."""
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if module is not self.vgae:
                module.train(mode)
        return self

    def forward(self,
                adj: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                lambda_: float = 1.0,
                mask: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full Stage-2 forward pass.

        **Flow:**

        1. **Stage-1 Inference**: Detached forward pass through VGAE to get $S_{pred}$.
        2. **Adaptive Routing**: Compute $\omega$ weights for channel-specific edges.
        3. **Disentangled Encoding**: Map features to C channels, shape $[N, C, H]$.
        4. **Adversarial Probing**: Use GRL and `demographic_classifier` to detect leakage.
        5. **Sensitivity Calculation**: Normalize channel losses by baseline entropy $H(s)$.
        6. **Feature Masking**: Apply `mask_mechanism` to sensitive channels. Skipped early for discriminator warmup.
        7. **Classification**: Predict target label $Y$ from the (masked) latent space.

        Parameters
        ----------
        adj : torch.Tensor
            Adjacency matrix [N, N].
        x : torch.Tensor
            Node features [N, D].
        edge_index : torch.Tensor
            Graph edges [2, E].
        lambda_ : float
            GRL scaling factor for adversarial training.
        mask : bool
            Whether to apply the masking mechanism.
            Defaults to True.

        Returns
        -------
        y_logits : torch.Tensor
            Target class predictions [N, num_classes].
        h : torch.Tensor
            The raw disentangled embeddings [N, C, H].
        c_logits : torch.Tensor
            Demographic predictions per channel [C, N, num_demographic_classes].
        s_pred : torch.Tensor
            The inferred sensitive labels from Stage 1.
        """
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
