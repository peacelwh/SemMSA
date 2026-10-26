"""Training objective: regression loss + L_csa + L_sep (+ optional regression loss on the semantic branch)."""
import torch
from torch import nn

from ..models.csa import spectral_alignment_losses


class SemMSALoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg['train']['loss']
        self.w_csa = c.get('w_csa', 1.0)
        self.w_sep = c.get('w_sep', 1.0)
        self.sigma = c.get('sigma', 1.0)
        self.tau = c.get('tau', 0.1)
        self.power_iters = c.get('power_iters', 20)
        self.w_aux_sem = c.get('w_aux_sem', 0.0)
        self.reg = nn.L1Loss() if c.get('reg_loss', 'mse') == 'l1' else nn.MSELoss()

    def forward(self, out, label):
        l_reg = self.reg(out['sentiment_preds'], label)
        l_csa, l_sep, _ = spectral_alignment_losses(out['V'], self.sigma, self.tau, self.power_iters)
        loss = l_reg + self.w_csa * l_csa + self.w_sep * l_sep
        logs = {'l_reg': l_reg.detach(), 'l_csa': l_csa.detach(), 'l_sep': l_sep.detach()}
        if self.w_aux_sem > 0 and 'aux_preds' in out:
            l_aux = self.reg(out['aux_preds'], label)
            loss = loss + self.w_aux_sem * l_aux
            logs['l_aux'] = l_aux.detach()
        logs['loss'] = loss
        return logs
