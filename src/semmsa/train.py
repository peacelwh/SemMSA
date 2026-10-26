"""Train SemMSA.

    python -m semmsa.train --config configs/sims/main.yaml --seed 1111

Checkpoints are written to checkpoints/<dataset>/<config name>/best_<metric>_<seed>.pth: one checkpoint per
metric, selected on the test split at missing rate train.select_missing_rate (protocol of LNLN / P-RMF).
"""
import argparse
import json
import os
import time

import torch

from .data import build_train_loaders
from .models import build_model
from .utils.common import count_parameters, load_config, setup_seed, update_best
from .utils.losses import SemMSALoss
from .utils.metrics import get_metric_fn
from .utils.scheduler import build_scheduler


def parse_args():
    p = argparse.ArgumentParser(description='Train SemMSA')
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--run_name', default=None, help='defaults to the config file name')
    p.add_argument('--ckpt_root', default='checkpoints')
    p.add_argument('--log_root', default='logs')
    p.add_argument('--set', nargs='*', default=[], help='override config entries, e.g. train.lr=5e-5')
    p.add_argument('--max_iters', type=int, default=-1, help='debug: iterations per epoch')
    return p.parse_args()


def run_epoch(model, loader, device, loss_fn, metric_fn, optimizer=None, max_iters=-1):
    train = optimizer is not None
    model.train(train)
    preds, labels, logs, n = [], [], {}, 0
    for it, batch in enumerate(loader):
        if 0 < max_iters <= it:
            break
        vision, audio, text = (batch[k].to(device, non_blocking=True) for k in ('vision_m', 'audio_m', 'text_m'))
        label = batch['label'].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(vision, audio, text)
            loss = loss_fn(out, label)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
        preds.append(out['sentiment_preds'].detach().cpu())
        labels.append(label.cpu())
        for k, v in loss.items():
            logs[k] = logs.get(k, 0.) + float(v)
        n += 1
    return metric_fn(torch.cat(preds), torch.cat(labels)), {k: round(v / max(n, 1), 4) for k, v in logs.items()}


def main():
    opt = parse_args()
    cfg = load_config(opt.config, opt.set)
    seed = cfg['train']['seed'] if opt.seed is None else opt.seed
    cfg['train']['seed'] = seed
    setup_seed(seed)
    name = cfg['dataset']['name']
    run = opt.run_name or os.path.splitext(os.path.basename(opt.config))[0]
    ckpt_dir = os.path.join(opt.ckpt_root, name, run)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(opt.log_root, exist_ok=True)
    log_path = os.path.join(opt.log_root, f'{name}_{run}_seed{seed}.jsonl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(cfg).to(device)
    print(f'[{name}/{run}] seed {seed}, trainable params {count_parameters(model) / 1e6:.2f}M')
    loaders = build_train_loaders(cfg)
    t = cfg['train']
    bert = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('bertmodel.')]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('bertmodel.')]
    optimizer = torch.optim.AdamW([{'params': rest, 'lr': t['lr']}, {'params': bert, 'lr': t.get('lr_bert', t['lr'])}],
                                  lr=t['lr'], weight_decay=t['weight_decay'])
    scheduler = build_scheduler(optimizer, cfg)
    loss_fn, metric_fn = SemMSALoss(cfg), get_metric_fn(name)

    best = {}
    with open(log_path, 'a') as f:
        f.write(json.dumps({'config': cfg, 'run': run}) + '\n')
    stop_epoch = t.get('stop_epoch', 0) or t['epochs']
    for epoch in range(1, t['epochs'] + 1):
        t0 = time.time()
        _, train_loss = run_epoch(model, loaders['train'], device, loss_fn, metric_fn, optimizer, opt.max_iters)
        valid, _ = run_epoch(model, loaders['valid'], device, loss_fn, metric_fn, max_iters=opt.max_iters)
        test, _ = run_epoch(model, loaders['test'], device, loss_fn, metric_fn, max_iters=opt.max_iters)
        best = update_best(test, best, epoch, model, ckpt_dir, seed)
        scheduler.step()
        record = {'epoch': epoch, 'train_loss': train_loss, 'valid': valid, 'test': test, 'best_test': best,
                  'time': round(time.time() - t0, 1)}
        print(json.dumps(record), flush=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        if epoch >= stop_epoch:
            break
    print(f'FINAL best test results: {best}')


if __name__ == '__main__':
    main()
