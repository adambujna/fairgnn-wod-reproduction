import numpy as np


class LossScheduler:
    def __init__(self, target_val, warmup_start, warmup_end, mode='linear'):
        self.target_val = target_val
        self.start = warmup_start
        self.end = warmup_end
        self.mode = mode

    def get_weight(self, epoch):
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
