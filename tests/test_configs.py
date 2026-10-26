import glob
import os
from types import SimpleNamespace

import pytest

from semmsa.evaluate import CHECKPOINT_OF, METRICS, build_plan
from semmsa.utils.common import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CONFIGS = sorted(c for c in glob.glob(os.path.join(ROOT, 'configs', '*', '*.yaml')) if not c.endswith('eval.yaml'))
DATASETS = ['mosi', 'mosei', 'sims']


@pytest.mark.parametrize('path', TRAIN_CONFIGS, ids=lambda p: os.path.relpath(p, ROOT))
def test_training_config(path):
    cfg = load_config(path)
    assert cfg['dataset']['name'] == os.path.basename(os.path.dirname(path))
    m = cfg['model']
    assert m['csr']['refine_steps'] == 4
    assert m['adapter']['num_prompts'] == 8 and m['adapter']['depth'] == 2
    assert len(m['feature_extractor']['input_dims']) == 3


def test_config_overrides():
    cfg = load_config(TRAIN_CONFIGS[0], ['train.lr=3e-4', 'train.loss.tau=0.5'])
    assert cfg['train']['lr'] == pytest.approx(3e-4)
    assert cfg['train']['loss']['tau'] == pytest.approx(0.5)


@pytest.mark.parametrize('dataset', DATASETS)
def test_eval_spec_covers_every_metric(dataset):
    spec = os.path.join(ROOT, 'configs', dataset, 'eval.yaml')
    name, seeds, plan = build_plan(SimpleNamespace(spec=spec, config=None, seeds=None))
    assert name == dataset
    assert seeds == [1111, 1112, 1113]
    assert set(plan) == set(METRICS[dataset])
    for metric, (config, key) in plan.items():
        assert os.path.isfile(config), config
        assert load_config(config)['dataset']['name'] == dataset
        assert key == CHECKPOINT_OF.get(metric, metric)
