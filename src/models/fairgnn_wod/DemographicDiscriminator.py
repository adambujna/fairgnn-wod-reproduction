#!/usr/bin/env/python
# Implementation of the demographic classifier and masking mechanism to
# obfuscate demographically relevant information in the disentangled embeddings.
#
# To our best knowledge, Wang et al. (2025) does not mention the classifier's architecture,
# so we decided to simply use a simple MLP just like the adaptive assigner.

import torch
import torch.nn as nn


class DemographicClassifier(nn.Module):
    """
    An MLP for sensitive attribute inference from latent channels.

    In the fGNN-WOD framework, this module acts as the discriminator.
    It is applied to each disentangled channel $h_c$ to estimate the leakage of demographic information.
    The resulting classification loss is used to calculate the `channel_sensitivity` score.

    Parameters
    ----------
    in_size : int
        Dimension of the input channel representation ($H$).
    hidden_size : int, optional
        Width of the internal hidden layer.
        Defaults to 32.
    num_classes : int, optional
        Number of output classes.
        Defaults to 2.
    """
    def __init__(self, in_size: int, hidden_size: int = 32, num_classes: int = 2):
        super(DemographicClassifier, self).__init__()
        # Shared MLP classifier across each channel
        # Input.shape = [N, H] (this is h_c)
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, h_c: torch.Tensor) -> torch.Tensor:
        """
        Predicts sensitive attributes from a specific latent channel.

        Parameters
        ----------
        h_c : torch.Tensor
            Latent features of a single channel, shape [N, H].

        Returns
        -------
        torch.Tensor
            Sensitive attribute logits of shape [N, num_classes].
        """
        return self.mlp(h_c)


class MaskMechanism(nn.Module):
    """
    Learns a soft feature mask to obfuscate sensitive information.

    The mask is parameterized as a learnable matrix $\mathbf{M} \in \mathbb{R}^{C \times H}$,
    passed through a sigmoid to constrain values to $[0, 1]$.
    It is applied selectively: channels with high `channel_sensitivity` are masked heavily,
    while neutral channels remain untouched.

    Parameters
    ----------
    in_size : int
        Dimension of the latent features per channel ($H$).
    num_channels : int
        Number of disentangled latent channels ($C$).
    init_value : float, optional
        Initial value for the mask parameters before sigmoid. A high value
        (e.g., 5.0) initializes the mask near 1.0 (no masking).
        Defaults to 5.0.
    """
    def __init__(self, in_size: int, num_channels: int, init_value: float = 5.0):
        super(MaskMechanism, self).__init__()
        self._init_value = init_value

        # Parameters are stored as [C, H] to allow unique masking per channel
        self.mask = nn.Parameter(torch.full((num_channels, in_size), init_value))
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        """Resets the mask weights to the initial constant value."""
        nn.init.constant_(self.mask, self._init_value)

    def forward(self, h: torch.Tensor, channel_sensitivity: torch.Tensor) -> torch.Tensor:
        """
        Applies the learned mask to latent representations based on sensitivity.

        The final mask for a channel is a linear interpolation:
        $m_{final} = (1 - \text{sens}) \cdot \mathbf{1} + \text{sens} \cdot \text{mask}$

        Parameters
        ----------
        h : torch.Tensor
            Disentangled latent representations of shape [N, C, H].
        channel_sensitivity : torch.Tensor
            Sensitivity scores for each channel, shape [C].
            Expected range $[0, 1]$.

        Returns
        -------
        torch.Tensor
            Masked latent representations of shape [N, C, H].
        """
        # Learnable component: [C, H]
        mask_h = 1.0 - self.sigmoid(self.mask)

        # Broadcast sensitivity across feature dimension: [C, 1]
        sens = channel_sensitivity.view(-1, 1)

        # If sens=0, m=1 (no mask). If sens=1, m=mask_h.
        m = (1.0 - sens) + (sens * mask_h)

        # Apply mask across all N nodes: [N, C, H] * [1, C, H]
        return h * m.unsqueeze(0)
