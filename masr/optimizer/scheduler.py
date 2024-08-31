import math
from typing import Union

import torch
try:
    from torch.optim.lr_scheduler import LRScheduler
except:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from typeguard import typechecked


class WarmupLR(LRScheduler):
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

    @typechecked
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: Union[int, float] = 25000,
            min_lr=1e-5,
            last_epoch: int = -1,
    ):
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


class NoamHoldAnnealing(LRScheduler):
    def __init__(self, optimizer, max_steps=175680, warmup_steps=None, warmup_ratio=0.2, hold_steps=None,
                 hold_ratio=0.3, decay_rate=1.0, min_lr=1.e-5, last_epoch=-1):
        """
        From Nemo:
        Implementation of the Noam Hold Annealing policy from the SqueezeFormer paper.

        Unlike NoamAnnealing, the peak learning rate can be explicitly set for this scheduler.
        The schedule first performs linear warmup,
        then holds the peak LR, then decays with some schedule for
        the remainder of the steps.
        Therefore the min-lr is still dependent on the hyper parameters selected.

        It's schedule is determined by three factors-

        Warmup Steps: Initial stage, where linear warmup
            occurs uptil the peak LR is reached. Unlike NoamAnnealing,
            the peak LR is explicitly stated here instead of a scaling factor.

        Hold Steps: Intermediate stage, where the peak LR
            is maintained for some number of steps. In this region,
            the high peak LR allows the model to converge faster
            if training is stable. However the high LR
            may also cause instability during training.
            Should usually be a significant fraction of training
            steps (around 30-40% of the entire training steps).

        Decay Steps: Final stage, where the LR rapidly decays
            with some scaling rate (set by decay rate).
            To attain Noam decay, use 0.5,
            for Squeezeformer recommended decay, use 1.0.
            The fast decay after prolonged high LR during
            hold phase allows for rapid convergence.

        References:
            - [Squeezeformer:
            An Efficient Transformer for Automatic Speech Recognition]
            (https://arxiv.org/abs/2206.00888)

        Args:
            optimizer: Pytorch compatible Optimizer object.
            warmup_steps: Number of training steps in warmup stage
            warmup_ratio: Ratio of warmup steps to total steps
            hold_steps: Number of training steps to hold the learning rate after warm up
            hold_ratio: Ratio of hold steps to total steps
            max_steps: Total number of steps while training or `None` for infinite training
            decay_rate: Float value describing the polynomial decay
                        after the hold period. Default value of 0.5 corresponds to Noam decay.
            min_lr: Minimum learning rate.
        """
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self._last_warmup_lr = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * max_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(optimizer, last_epoch)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def get_lr(self):
        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return self.base_lrs

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    @staticmethod
    def _noam_hold_annealing(initial_lr, step, warmup_steps, hold_steps, decay_rate, min_lr):
        # hold_steps = total number of steps
        # to hold the LR, not the warmup + hold steps.
        T_warmup_decay = max(1, warmup_steps ** decay_rate)
        T_hold_decay = max(1, (step - hold_steps) ** decay_rate)
        lr = (initial_lr * T_warmup_decay) / T_hold_decay
        lr = max(lr, min_lr)
        return lr

    def _get_lr(self, step):
        if self.warmup_steps is None or self.warmup_steps == 0:
            raise ValueError("Noam scheduler cannot be used without warmup steps")

        if self.hold_steps > 0:
            hold_steps = self.hold_steps - self.warmup_steps
        else:
            hold_steps = 0

        new_lrs = [
            self._noam_hold_annealing(initial_lr=initial_lr,
                                      step=step,
                                      warmup_steps=self.warmup_steps,
                                      hold_steps=hold_steps,
                                      decay_rate=self.decay_rate,
                                      min_lr=self.min_lr)
            for initial_lr in self.base_lrs
        ]
        return new_lrs

    def set_step(self, step: int):
        self.last_epoch = step


class CosineWithWarmup(LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, T_max, eta_min=0, warmup_steps=None, warmup_ratio=0.2,
                 last_epoch=-1):
        """
        Set the learning rate using a cosine annealing schedule, where :math:`\eta_{max}` is set to
        the initial learning_rate.

        Args:
            optimizer: Pytorch compatible Optimizer object.
            T_max (int): Maximum number of iterations. It is half of the decay cycle of learning rate. It must be a positive integer.
            eta_min (float|int, optional): Minimum learning rate. Default: 0.
            warmup_steps (int): Number of training steps in warmup stage
            warmup_ratio (float): Ratio of warmup steps to total steps
            last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        Returns:
            ``CosineWithWarmup`` instance to schedule learning rate.
        """
        if not isinstance(T_max, int):
            raise TypeError("The type of 'T_max' must be 'int', but received %s." % type(T_max))
        if not isinstance(eta_min, (float, int)):
            raise TypeError("The type of 'eta_min' must be 'float, int', but received %s." % type(eta_min))
        assert T_max > 0 and isinstance(T_max, int), " 'T_max' must be a positive integer."
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * T_max)
        else:
            self.warmup_steps = 0
        self.T_max = T_max - self.warmup_steps
        self.eta_min = float(eta_min)
        super(CosineWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        return self._get_lr(step - self.warmup_steps)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_lr(self, step):
        if (step - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]

        return [(1 + math.cos(math.pi * step / self.T_max)) /
                (1 + math.cos(math.pi * (step - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups]

    def set_step(self, step: int):
        self.last_epoch = step
