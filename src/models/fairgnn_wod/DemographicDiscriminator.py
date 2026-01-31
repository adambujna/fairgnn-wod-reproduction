#!/usr/bin/env/python
# Implementation of the demographic classifier and masking mechanism to
# obfuscate demographically relevant information in the disentangled embeddings.
#
# To my best knowledge, Wang et al. (2025) does not mention the classifier's architecture,
# so we decided to simply use a simple MLP just like the adaptive assigner.

import torch
import torch.nn as nn


class DemographicClassifier(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(DemographicClassifier, self).__init__()
        # Shared MLP classifier across each channel
        # Input.shape = [N, H] (this is h_c)
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, h_c):     # [N, H]
        out = self.mlp(h_c)     # [N, 2]
        return out


class MaskMechanism(nn.Module):
    def __init__(self, in_size, num_channels, init_value=5.0):
        super(MaskMechanism, self).__init__()
        self._init_value = init_value

        self.mask = nn.Parameter(torch.ones((num_channels, in_size)) * init_value)   # [C, H] applied to each N
        self.sigmoid = nn.Sigmoid()     # To constrain mask to [0,1]

    def reset_parameters(self):
        nn.init.constant_(self.mask, self._init_value)

    def forward(self, h, channel_sensitivity):     # [N, C, H], [C]
        mask_h = 1.0 - self.sigmoid(self.mask)   # [C, H]

        # Selectively apply mask only if a channel is sensitive
        sens = channel_sensitivity.view(-1, 1)
        m = (1.0 - sens) + (sens * mask_h)
        return h * m.unsqueeze(0)   # [N, C, H] * [1, C, H]
