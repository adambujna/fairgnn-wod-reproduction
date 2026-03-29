#!/usr/bin/env/python

import numpy as np


class LossScheduler:
    """
    Manages the dynamic weighting of loss components during training.

    Used for 'warm-up' periods where specific penalties (like independence or fairness losses)
    are gradually introduced to allow the model to first learn basic structural or predictive features.

    Parameters
    ----------
    target_val : float
        The final maximum weight the loss component will reach.
    warmup_start : int
        The epoch at which the weight starts increasing from 0.0.
    warmup_end : int
        The epoch at which the weight reaches `target_val`.
    mode : str, optional
        The interpolation strategy: 'linear' or 'sigmoid'.
        Defaults to 'linear'.
    """
    def __init__(self, target_val: float, warmup_start: int, warmup_end: int, mode: str = 'linear'):
        self.target_val = target_val
        self.start = warmup_start
        self.end = warmup_end
        self.mode = mode

    def get_weight(self, epoch: int) -> float:
        """
        Calculates the loss weight for a given epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch.

        Returns
        -------
        float
            The calculated weight $\lambda \in [0, \text{target\_val}]$.
        """
        if epoch < self.start:
            return 0.0
        if epoch >= self.end:
            return self.target_val

        progress = (epoch - self.start) / (self.end - self.start)

        if self.mode == 'linear':
            return self.target_val * progress

        if self.mode == 'sigmoid':
            return self.target_val * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)

        return self.target_val
