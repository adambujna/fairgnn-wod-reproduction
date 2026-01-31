#!/usr/bin/env/python

import torch
import warnings


class EarlyStoppingCriterion:
    def __init__(self, patience=20, best_delta=0.0, mode='min', start_from_epoch=0, save_basename=None):
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

    def step(self, val_score, model=None):
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

    def is_improvement(self, val_score):
        if self.mode == 'min':
            return val_score < (self.best_score - self.delta)
        else:   # max
            return val_score > (self.best_score + self.delta)
