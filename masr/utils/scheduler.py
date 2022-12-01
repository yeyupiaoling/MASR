from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler

from typeguard import check_argument_types


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: Union[int, float] = 25000,
            min_lr=1e-5,
            last_epoch: int = -1,
    ):
        assert check_argument_types()
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, lr={self.base_lr}, min_lr={self.min_lr}, last_epoch={self.last_epoch})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            lrs = []
            for lr in self.base_lrs:
                lr = lr * step_num ** -0.5
                if lr < self.min_lr:
                    lr = self.min_lr
                lrs.append(lr)
            return lrs
        else:
            lrs = []
            for lr in self.base_lrs:
                lr = lr * self.warmup_steps ** 0.5 * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
                if lr < self.min_lr and step_num > self.warmup_steps:
                    lr = self.min_lr
                lrs.append(lr)
            return lrs

    def set_step(self, step: int):
        self.last_epoch = step
