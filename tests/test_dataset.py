"""Missing-data protocol on a small synthetic dataset in the processed MMSA format."""
import numpy as np
import pytest

from semmsa.data.dataset import UNK_ID, MMDataset

N, L, TA, TV = 12, 10, 6, 5


def make_split(rng):
    lengths = rng.integers(4, L + 1, N)
    text = np.zeros((N, 3, L), dtype=np.float32)
    for i, n in enumerate(lengths):
        text[i, 0, :n] = np.concatenate(([101], rng.integers(1000, 2000, n - 2), [102]))
        text[i, 1, :n] = 1
    audio = rng.normal(size=(N, TA, 3)).astype(np.float32)
    vision = rng.normal(size=(N, TV, 4)).astype(np.float32)
    return {'text_bert': text, 'audio': audio, 'vision': vision,
            'regression_labels': rng.uniform(-3, 3, N).astype(np.float32), 'id': [str(i) for i in range(N)],
            'audio_lengths': [TA] * N, 'vision_lengths': [TV] * N}


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    return {s: make_split(rng) for s in ('train', 'valid', 'test')}


def cfg():
    return {'dataset': {'name': 'mosi', 'data_path': None},
            'train': {'seed': 1111, 'select_missing_rate': 0.5, 'batch_size': 4, 'num_workers': 0}}


def test_no_missing_keeps_inputs(data):
    ds = MMDataset(cfg(), 'test', data=data, missing_rate=0.0, seed=1111)
    for i in range(N):
        item = ds[i]
        assert np.array_equal(item['text_m'][0].numpy(), data['test']['text_bert'][i, 0])
        assert np.allclose(item['audio_m'].numpy(), data['test']['audio'][i])
        assert np.allclose(item['vision_m'].numpy(), data['test']['vision'][i])


def test_full_missing_uses_unk_and_zeros(data):
    ds = MMDataset(cfg(), 'test', data=data, missing_rate=1.0, seed=1111)
    for i in range(N):
        item = ds[i]
        ids, mask = item['text_m'][0].numpy(), data['test']['text_bert'][i, 1]
        n = int(mask.sum())
        assert ids[0] == 101 and ids[n - 1] == 102  # [CLS] / [SEP] are kept
        assert (ids[1:n - 1] == UNK_ID).all()
        assert (ids[n:] == 0).all()  # padding is untouched
        assert (item['audio_m'] == 0).all() and (item['vision_m'] == 0).all()


def test_test_masks_are_deterministic_per_seed(data):
    a = MMDataset(cfg(), 'test', data=data, missing_rate=0.5, seed=1111)
    b = MMDataset(cfg(), 'test', data=data, missing_rate=0.5, seed=1111)
    c = MMDataset(cfg(), 'test', data=data, missing_rate=0.5, seed=1112)
    assert np.array_equal(a.audio_missing_mask, b.audio_missing_mask)
    assert np.array_equal(a.text_missing_mask, b.text_missing_mask)
    assert not np.array_equal(a.audio_missing_mask, c.audio_missing_mask)


def test_half_of_training_instances_complete(data):
    ds = MMDataset(cfg(), 'train', data=data, seed=1111)
    for keep, valid in ((ds.audio_missing_mask, None), (ds.vision_missing_mask, None), (ds.text_missing_mask, ds.text_mask)):
        valid = np.ones_like(keep) if valid is None else valid
        complete = (keep == valid).all(axis=1)
        assert complete.sum() >= N // 2
