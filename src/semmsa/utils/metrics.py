"""Evaluation metrics, following the evaluation code of P-RMF (ACL 2025) / LNLN (NeurIPS 2024)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _multiclass_acc(y_pred, y_true):
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


def eval_mosi_mosei(y_pred, y_true):
    preds = y_pred.view(-1).cpu().detach().numpy()
    truth = y_true.view(-1).cpu().detach().numpy()
    mae = np.mean(np.absolute(preds - truth))
    corr = np.corrcoef(preds, truth)[0][1]
    acc7 = _multiclass_acc(np.clip(preds, -3., 3.), np.clip(truth, -3., 3.))
    acc5 = _multiclass_acc(np.clip(preds, -2., 2.), np.clip(truth, -2., 2.))

    non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])
    nz_truth, nz_preds = truth[non_zeros] > 0, preds[non_zeros] > 0
    bin_truth, bin_preds = truth >= 0, preds >= 0
    return {
        'Has0_acc_2': round(float(accuracy_score(bin_preds, bin_truth)), 4),
        'Has0_F1_score': round(float(f1_score(bin_preds, bin_truth, average='weighted')), 4),
        'Non0_acc_2': round(float(accuracy_score(nz_preds, nz_truth)), 4),
        'Non0_F1_score': round(float(f1_score(nz_preds, nz_truth, average='weighted')), 4),
        'Mult_acc_5': round(float(acc5), 4),
        'Mult_acc_7': round(float(acc7), 4),
        'MAE': round(float(mae), 4),
        'Corr': round(float(corr), 4),
    }


def _bins(x, edges):
    out = x.copy()
    for i in range(len(edges) - 1):
        out[np.logical_and(x > edges[i], x <= edges[i + 1])] = i
    return out


def eval_sims(y_pred, y_true):
    preds = np.clip(y_pred.view(-1).cpu().detach().numpy(), -1., 1.)
    truth = np.clip(y_true.view(-1).cpu().detach().numpy(), -1., 1.)
    ms_2, ms_3, ms_5 = [-1.01, 0.0, 1.01], [-1.01, -0.1, 0.1, 1.01], [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
    p2, t2 = _bins(preds, ms_2), _bins(truth, ms_2)
    return {
        'Mult_acc_2': round(float(_multiclass_acc(p2, t2)), 4),
        'Mult_acc_3': round(float(_multiclass_acc(_bins(preds, ms_3), _bins(truth, ms_3))), 4),
        'Mult_acc_5': round(float(_multiclass_acc(_bins(preds, ms_5), _bins(truth, ms_5))), 4),
        'F1_score': round(float(f1_score(p2, t2, average='weighted')), 4),
        'MAE': round(float(np.mean(np.absolute(preds - truth))), 4),
        'Corr': round(float(np.corrcoef(preds, truth)[0][1]), 4),
    }


def get_metric_fn(dataset_name):
    return eval_sims if dataset_name.lower() == 'sims' else eval_mosi_mosei
