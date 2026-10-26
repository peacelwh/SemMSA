"""Linear warm-up followed by cosine annealing (as in LNLN)."""
import torch
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if not self.finished:
                self.after_scheduler.base_lrs = list(self.base_lrs)
                self.finished = True
            return self.after_scheduler.get_last_lr()
        return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            self.after_scheduler.step(None)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


def build_scheduler(optimizer, cfg):
    epochs = cfg['train']['epochs']
    warmup = cfg['train'].get('warmup_ratio', 0.1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(1 - warmup) * epochs)
    return GradualWarmupScheduler(optimizer, total_epoch=warmup * epochs, after_scheduler=cosine)
