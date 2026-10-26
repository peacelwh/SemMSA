import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from semmsa.utils.metrics import eval_mosi_mosei, eval_sims, get_metric_fn


def test_mosi_perfect_predictions():
    y = torch.tensor([-3.0, -1.2, -0.4, 0.0, 0.6, 1.4, 2.2, 3.0])
    res = eval_mosi_mosei(y.clone(), y)
    for k in ('Has0_acc_2', 'Has0_F1_score', 'Non0_acc_2', 'Non0_F1_score', 'Mult_acc_5', 'Mult_acc_7', 'Corr'):
        assert res[k] == pytest.approx(1.0)
    assert res['MAE'] == pytest.approx(0.0)


def test_mosi_has0_and_non0():
    truth = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    preds = torch.tensor([-1.5, 0.5, -0.5, 1.2, 1.8])
    res = eval_mosi_mosei(preds, truth)
    # Has0: negative vs non-negative over all 5 samples -> 3 correct
    assert res['Has0_acc_2'] == pytest.approx(0.6)
    # Non0: zero labels removed, negative vs positive -> 3 of 4 correct
    assert res['Non0_acc_2'] == pytest.approx(0.75)


def test_f1_is_weighted():
    rng = np.random.default_rng(0)
    truth = torch.tensor(rng.uniform(-3, 3, 200), dtype=torch.float32)
    preds = torch.tensor(np.clip(truth.numpy() * 0.3 + 0.8, -3, 3), dtype=torch.float32)  # biased to positive
    res = eval_mosi_mosei(preds, truth)
    p, t = preds.numpy() >= 0, truth.numpy() >= 0
    assert res['Has0_F1_score'] == pytest.approx(round(f1_score(p, t, average='weighted'), 4))


def test_sims_metrics():
    truth = torch.tensor([-1.0, -0.6, -0.2, 0.0, 0.2, 0.8, 1.0])
    res = eval_sims(truth.clone(), truth)
    for k in ('Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'F1_score', 'Corr'):
        assert res[k] == pytest.approx(1.0)
    assert res['MAE'] == pytest.approx(0.0)
    assert get_metric_fn('sims') is eval_sims
    assert get_metric_fn('mosei') is eval_mosi_mosei
