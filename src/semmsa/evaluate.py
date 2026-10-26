"""Robustness evaluation under intra-modal missing rates r = 0.0, 0.1, ..., 0.9.

    python -m semmsa.evaluate --spec configs/sims/eval.yaml            # paper protocol, one checkpoint per metric
    python -m semmsa.evaluate --config configs/sims/main.yaml           # all metrics from one training config

Every metric is computed with its own best checkpoint listed in configs/<dataset>/eval.yaml; F1 and Corr use the
checkpoints of Acc-2 and MAE.
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from .data import build_eval_loader, load_pickle
from .models import build_model
from .utils.common import load_checkpoint, load_config
from .utils.metrics import get_metric_fn

RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = {
    'mosi': ['Has0_acc_2', 'Has0_F1_score', 'Non0_acc_2', 'Non0_F1_score', 'Mult_acc_5', 'Mult_acc_7', 'MAE', 'Corr'],
    'sims': ['Mult_acc_2', 'F1_score', 'Mult_acc_3', 'Mult_acc_5', 'MAE', 'Corr'],
}
METRICS['mosei'] = METRICS['mosi']
CHECKPOINT_OF = {'Has0_F1_score': 'Has0_acc_2', 'Non0_F1_score': 'Non0_acc_2', 'F1_score': 'Mult_acc_2', 'Corr': 'MAE'}
FIRST_SEED_ONLY = {'mosi': {'Has0_acc_2', 'Has0_F1_score', 'Non0_acc_2', 'Non0_F1_score', 'Mult_acc_5'}, 'sims': set()}
FIRST_SEED_ONLY['mosei'] = FIRST_SEED_ONLY['mosi']
NAMES = {'Has0_acc_2': 'Acc-2 (Has0)', 'Has0_F1_score': 'F1 (Has0)', 'Non0_acc_2': 'Acc-2 (Non0)', 'Non0_F1_score': 'F1 (Non0)',
         'Mult_acc_2': 'Acc-2', 'F1_score': 'F1', 'Mult_acc_3': 'Acc-3', 'Mult_acc_5': 'Acc-5', 'Mult_acc_7': 'Acc-7',
         'MAE': 'MAE', 'Corr': 'Corr'}


def parse_args():
    p = argparse.ArgumentParser(description='Robust evaluation of SemMSA')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--spec', help='evaluation spec (configs/<dataset>/eval.yaml)')
    g.add_argument('--config', help='evaluate all metrics with the checkpoints of one training config')
    p.add_argument('--seeds', type=int, nargs='*', default=None)
    p.add_argument('--ckpt_root', default='checkpoints')
    p.add_argument('--out_dir', default='results')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, metric_fn, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        out = model(batch['vision_m'].to(device), batch['audio_m'].to(device), batch['text_m'].to(device))
        preds.append(out['sentiment_preds'].float().cpu())
        labels.append(batch['label'])
    return metric_fn(torch.cat(preds), torch.cat(labels))


def build_plan(opt):
    """-> dataset, seeds, {metric: (config path, checkpoint key)}."""
    if opt.spec:
        with open(opt.spec) as f:
            spec = yaml.safe_load(f)
        root = os.path.dirname(opt.spec)
        plan = {m: (os.path.join(root, v['config'] + '.yaml'), v['checkpoint']) for m, v in spec['metrics'].items()}
        return spec['dataset'], opt.seeds or spec['seeds'], plan
    name = load_config(opt.config)['dataset']['name']
    plan = {m: (opt.config, CHECKPOINT_OF.get(m, m)) for m in METRICS[name]}
    return name, opt.seeds or [1111, 1112, 1113], plan


def main():
    opt = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, seeds, plan = build_plan(opt)
    metric_fn = get_metric_fn(dataset)
    data = None
    per_seed = {}  # (config, ckpt key) -> seed -> rate -> metrics
    for config in sorted({c for c, _ in plan.values()}):
        cfg = load_config(config)
        data = data if data is not None else load_pickle(cfg['dataset']['data_path'])
        run = os.path.splitext(os.path.basename(config))[0]
        model = build_model(cfg).to(device)
        for key in sorted({k for c, k in plan.values() if c == config}):
            users = [m for m, (c, k) in plan.items() if c == config and k == key]
            needed = seeds[:1] if all(m in FIRST_SEED_ONLY[dataset] for m in users) else seeds
            for seed in needed:
                ckpt = os.path.join(opt.ckpt_root, dataset, run, f'best_{key}_{seed}.pth')
                load_checkpoint(model, ckpt, device)
                res = {}
                for r in RATES:
                    res[r] = evaluate(model, build_eval_loader(cfg, r, seed, data=data), metric_fn, device)
                per_seed.setdefault((config, key), {})[seed] = res
                print(f'{run}/best_{key}_{seed}: r=0.0 {res[0.0]}', flush=True)
        del model
        torch.cuda.empty_cache()

    summary = {}
    for metric, (config, key) in plan.items():
        use = seeds[:1] if metric in FIRST_SEED_ONLY[dataset] else seeds
        curve = [float(np.mean([per_seed[(config, key)][s][r][metric] for s in use])) for r in RATES]
        summary[metric] = {'per_rate': curve, 'avg': float(np.mean(curve)), 'seeds': use,
                           'config': os.path.basename(config), 'checkpoint': key}

    os.makedirs(opt.out_dir, exist_ok=True)
    scale = lambda m, v: f'{v:.3f}' if m in ('MAE', 'Corr') else f'{v * 100:.2f}'
    lines = ['| Metric | ' + ' | '.join(f'r={r}' for r in RATES) + ' | Avg |', '|' + '---|' * (len(RATES) + 2)]
    for m in METRICS[dataset]:
        if m in summary:
            s = summary[m]
            lines.append(f'| {NAMES[m]} | ' + ' | '.join(scale(m, v) for v in s['per_rate']) + f' | {scale(m, s["avg"])} |')
    md = '\n'.join(lines)
    print('\n' + md)
    with open(os.path.join(opt.out_dir, f'{dataset}_results.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(opt.out_dir, f'{dataset}_results.md'), 'w') as f:
        f.write(md + '\n')


if __name__ == '__main__':
    main()
