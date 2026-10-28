import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from sepsis6q.evaluate import (ScoreIndex, auprc, auroc, operating_point, patient_bootstrap_weights,
                               sens_at_fpr, threshold_at_specificity)


def _data(seed, ties=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 3000)
    s = rng.normal(size=3000) + 0.8 * y
    if ties:
        s = np.round(s, 1)
    return s, y


def test_auroc_and_auprc_match_sklearn():
    for ties in (False, True):
        s, y = _data(1, ties)
        pos, neg = ScoreIndex(s, y).tallies()
        assert abs(auroc(pos, neg) - roc_auc_score(y, s)) < 1e-12
        assert abs(auprc(pos, neg) - average_precision_score(y, s)) < 1e-12


def test_weighted_metrics_match_sklearn():
    s, y = _data(2, ties=True)
    w = np.random.default_rng(3).integers(0, 4, len(y)).astype(float)
    pos, neg = ScoreIndex(s, y).tallies(w)
    assert abs(auroc(pos, neg) - roc_auc_score(y, s, sample_weight=w)) < 1e-12
    assert abs(auprc(pos, neg) - average_precision_score(y, s, sample_weight=w)) < 1e-12


def test_sensitivity_at_fpr_interpolates_sklearn_roc():
    s, y = _data(4)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    pos, neg = ScoreIndex(s, y).tallies()
    assert abs(sens_at_fpr(pos, neg, 0.05) - np.interp(0.05, fpr, tpr)) < 1e-12


def test_threshold_reaches_specificity():
    s, y = _data(5, ties=True)
    for spec in (0.9, 0.95):
        thr = threshold_at_specificity(s, y, spec)
        assert np.mean(s[y == 0] < thr) >= spec
        op = operating_point(s, y, np.arange(len(y)), y, thr)
        assert op["false_alarm_rate"] <= 1 - spec + 1e-12


def test_bootstrap_weights_resample_whole_patients():
    patients = np.repeat(np.arange(50), 4)
    for w in patient_bootstrap_weights(patients, 5, 0):
        assert w.sum() == len(patients)
        assert np.all(w.reshape(50, 4) == w.reshape(50, 4)[:, :1])
