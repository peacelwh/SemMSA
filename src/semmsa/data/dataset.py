"""MOSI / MOSEI / SIMS with intra-modal random missingness (protocol of LNLN and P-RMF).

Missing visual and acoustic frames are replaced by zero vectors and missing language tokens by [UNK].
Training instances draw a missing rate from U(0, 1) for every modality, with 50% of the instances kept complete
per modality; test instances use a fixed missing rate r.  `train_missing_shared` > 0 lets that fraction of the
training instances share one missing rate across the three modalities.
"""
import gc
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

UNK_ID = 100
_PICKLE_CACHE = {}


def load_pickle(path):
    """Load a processed MMSA pickle once per process (MOSEI is about 13 GB)."""
    if path not in _PICKLE_CACHE:
        with open(path, 'rb') as f:
            _PICKLE_CACHE[path] = pickle.load(f)
    return _PICKLE_CACHE[path]


class MMDataset(Dataset):
    def __init__(self, cfg, split='train', data=None, missing_rate=None, seed=None):
        self.split = split
        self.seed = cfg['train']['seed'] if seed is None else seed
        self.eval_missing_rate = cfg['train']['select_missing_rate'] if missing_rate is None else missing_rate
        self.train_missing_shared = float(cfg['dataset'].get('train_missing_shared', 0.0))
        data = load_pickle(cfg['dataset']['data_path']) if data is None else data
        d = data[split]

        self.text = d['text_bert'].astype(np.float32)
        self.vision = d['vision'].astype(np.float32)
        self.audio = d['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.labels = d['regression_labels'].astype(np.float32)
        self.ids = list(d['id'])
        self.audio_lengths = list(d['audio_lengths'])
        self.vision_lengths = list(d['vision_lengths'])

        n = len(self.labels)
        rates = self._sample_train_rates(n) if split == 'train' else [self.eval_missing_rate * np.ones((n, 1)) for _ in range(3)]
        self._apply_missing(rates)

    def _sample_train_rates(self, n):
        rates = [np.random.uniform(size=(n, 1)) for _ in range(3)]
        for i in range(3):
            rates[i][random.sample(list(range(n)), int(n / 2))] = 0
        if self.train_missing_shared > 0:
            idx = random.sample(list(range(n)), int(n * self.train_missing_shared))
            shared = np.random.uniform(size=(len(idx), 1))
            for i in range(3):
                rates[i][idx] = shared
        return rates

    def _apply_missing(self, rates):
        self.text_mask, self.text_missing_mask = self._masks(rates[0], mode='text')
        self.audio_missing_mask = self._masks(rates[1], mode='audio')[1]
        self.vision_missing_mask = self._masks(rates[2], mode='vision')[1]

    def _masks(self, rate, mode):
        if mode == 'text':
            input_mask = np.asarray(self.text[:, 1, :], dtype=np.float32)
            lengths = np.argmin(input_mask, axis=1)
        else:
            lengths = self.audio_lengths if mode == 'audio' else self.vision_lengths
            T = (self.audio if mode == 'audio' else self.vision).shape[1]
            input_mask = (np.arange(T)[None, :] < np.asarray(lengths)[:, None]).astype(np.float32)
        np.random.seed(self.seed)
        keep = ((np.random.uniform(size=input_mask.shape) > rate.repeat(input_mask.shape[1], 1)) * input_mask).astype(np.float32)
        if mode == 'text':  # [CLS] and [SEP] are never missing
            rows = np.arange(len(keep))
            keep[rows, 0] = 1
            keep[rows, lengths - 1] = 1
        return input_mask, keep

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.split == 'train' and index == 0:  # new missing pattern every epoch
            self._apply_missing(self._sample_train_rates(len(self.labels)))
        ids, keep, im = self.text[index, 0, :], self.text_missing_mask[index], self.text_mask[index]
        text_m = np.stack((keep * ids + UNK_ID * (im - keep), im, self.text[index, 2, :]), axis=0).astype(np.float32)
        return {
            'text_m': torch.from_numpy(text_m),
            'audio_m': torch.from_numpy(self.audio[index] * self.audio_missing_mask[index][:, None]),
            'vision_m': torch.from_numpy(self.vision[index] * self.vision_missing_mask[index][:, None]),
            'label': torch.from_numpy(np.asarray(self.labels[index], dtype=np.float32).reshape(-1)),
        }


def build_train_loaders(cfg):
    path = cfg['dataset']['data_path']
    data = load_pickle(path)
    splits = {s: MMDataset(cfg, s, data=data) for s in ('train', 'valid', 'test')}
    _PICKLE_CACHE.pop(path, None)  # the splits keep float32 copies; release the raw pickle
    del data
    gc.collect()
    t = cfg['train']
    return {s: DataLoader(ds, batch_size=t['batch_size'], num_workers=t['num_workers'], shuffle=(s == 'train'),
                          pin_memory=True, drop_last=False) for s, ds in splits.items()}


def build_eval_loader(cfg, missing_rate, seed, data=None):
    ds = MMDataset(cfg, 'test', data=data, missing_rate=missing_rate, seed=seed)
    return DataLoader(ds, batch_size=cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'],
                      shuffle=False, pin_memory=True)
