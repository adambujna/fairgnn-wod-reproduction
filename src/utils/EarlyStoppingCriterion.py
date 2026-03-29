#!/usr/bin/env/python

import torch
import warnings


class EarlyStoppingCriterion:
    """
    Monitors a validation metric to stop training when performance plateaus.

    This class tracks the 'best' version of the model and automatically
    saves checkpoints to disk when an improvement is detected.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait for an improvement before stopping.
        Defaults to 20.
    best_delta : float, optional
        Minimum change in the monitored score to qualify as an improvement.
        Defaults to 0.0.
    mode : str, optional
        Whether to monitor for a minimum ('min', e.g., loss) or maximum ('max', e.g., accuracy).
        Defaults to 'min'.
    start_from_epoch : int, optional
        The epoch after which the monitoring begins. Defaults to 0.
    save_basename : str, optional
        Path and filename prefix for saving the best model checkpoint.
    """
    def __init__(self,
                 patience: int = 20,
                 best_delta: float = 0.0,
                 mode: str = 'min',
                 start_from_epoch: int = 0,
                 save_basename: bool = None):
        self.start_from_epoch = start_from_epoch
        self.patience = patience
        self.delta = best_delta
        self.mode = mode
        self.counter = 0
        self.epoch = 0
        self.best_score = None
        self.early_stop = False

        self.save_basename = save_basename
        if self.save_basename is None:
            warnings.warn("did not get model save directory + file name", Warning)

    def step(self, val_score: float, model: torch.nn.Module = None) -> bool:
        """
        Updates the internal state with a new validation score.

        Parameters
        ----------
        val_score : float
            The current metric value (e.g., Validation Loss or F1-score).
        model : torch.nn.Module, optional
            The model to save if a new best score is achieved.

        Returns
        -------
        bool
            True if the early stopping condition has been met, False otherwise.
        """
        self.epoch += 1
        if self.epoch < self.start_from_epoch:
            return False
        # Determine if the score is "better" based on mode
        if self.best_score is None:
            self.best_score = val_score
            if model is not None:
                if self.save_basename is None:
                    self.save_basename = model.__class__.__name__
                torch.save(model.state_dict(), self.save_basename + f"_best.pt")
        elif self.is_improvement(val_score):
            self.best_score = val_score
            self.counter = 0
            if model is not None:
                if self.save_basename is None:
                    self.save_basename = model.__class__.__name__
                torch.save(model.state_dict(), self.save_basename + f"_best.pt")
        else:
            self.counter += 1
        if self.counter >= self.patience and self.epoch >= self.start_from_epoch + self.patience:
            self.early_stop = True
        return self.early_stop

    def is_improvement(self, val_score: float) -> bool:
        """Checks if the new score outperforms the current best score."""
        if self.mode == 'min':
            return val_score < (self.best_score - self.delta)
        else:   # max
            return val_score > (self.best_score + self.delta)
