import os
import random

import numpy as np
import torch
import yaml

# metrics with their own best checkpoint (selected on the test split at r = select_missing_rate)
HIGHER_BETTER = ['Has0_acc_2', 'Non0_acc_2', 'Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'Mult_acc_7', 'Corr']
LOWER_BETTER = ['MAE']
PAIRED_F1 = {'Has0_acc_2': 'Has0_F1_score', 'Non0_acc_2': 'Non0_F1_score', 'Mult_acc_2': 'F1_score'}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        key, value = kv.split('=', 1)
        d = cfg
        parts = key.split('.')
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = _parse_value(value)
    return cfg


def _parse_value(value):
    v = yaml.safe_load(value)
    if isinstance(v, str):  # YAML 1.1 reads '3e-4' as a string
        try:
            return float(v)
        except ValueError:
            pass
    return v


def trainable_state_dict(model):
    """State dict without frozen weights (the LLM and, if frozen, BERT) to keep checkpoints small."""
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k not in frozen and not k.startswith('csr.llm.')}


def save_checkpoint(path, epoch, model):
    torch.save({'epoch': epoch, 'state_dict': trainable_state_dict(model)}, path)


def load_checkpoint(model, path, device='cpu'):
    states = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(states['state_dict'], strict=False)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    missing = [m for m in missing if not m.startswith('csr.llm.') and m not in frozen and not m.endswith('position_ids')]
    if missing or unexpected:
        raise RuntimeError(f'checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    return states


def update_best(results, best, epoch, model, ckpt_dir, seed, save=True):
    """Track the best value of every metric and save one checkpoint per metric (LNLN protocol)."""
    improved = []
    if not best:
        best.update(results)
        best['_epoch'] = {k: epoch for k in results}
        improved = [k for k in results if k in HIGHER_BETTER + LOWER_BETTER]
    else:
        for key, value in results.items():
            prev = best[key]
            if value != value:  # NaN never wins
                continue
            stale = prev != prev
            if (key in HIGHER_BETTER and (stale or value > prev)) or (key in LOWER_BETTER and (stale or value < prev)):
                best[key] = value
                best['_epoch'][key] = epoch
                if key in PAIRED_F1:
                    best[PAIRED_F1[key]] = results[PAIRED_F1[key]]
                improved.append(key)
    if save:
        for key in improved:
            save_checkpoint(os.path.join(ckpt_dir, f'best_{key}_{seed}.pth'), epoch, model)
    return best


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
