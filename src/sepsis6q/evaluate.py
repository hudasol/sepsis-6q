"""Metrics, operating points and the patient level bootstrap.

All ranking metrics are computed from per score tallies of positive and negative
weight, so that ties are handled exactly and bootstrap resamples (which are
patient weights) reuse a single sort.
"""

import numpy as np

from .config import TARGET_FPR


class ScoreIndex:
    """Sorted unique scores of one evaluation set, reused for every resample."""

    def __init__(self, scores, y):
        scores = np.asarray(scores, dtype=float)
        self.y = np.asarray(y).astype(bool)
        uniq, inv = np.unique(-scores, return_inverse=True)  # descending score order
        self.inv = inv.ravel()
        self.n_unique = len(uniq)

    def tallies(self, w=None):
        w = np.ones(len(self.y)) if w is None else w
        pos = np.bincount(self.inv, weights=w * self.y, minlength=self.n_unique)
        neg = np.bincount(self.inv, weights=w * ~self.y, minlength=self.n_unique)
        return pos, neg


def _roc_points(pos, neg):
    tp = np.concatenate([[0.0], np.cumsum(pos)])
    fp = np.concatenate([[0.0], np.cumsum(neg)])
    return tp / tp[-1], fp / fp[-1]


def sens_at_fpr(pos, neg, fpr_target=TARGET_FPR):
    """Sensitivity at exactly the target false positive rate, by linear interpolation on the ROC curve."""
    tpr, fpr = _roc_points(pos, neg)
    k = int(np.searchsorted(fpr, fpr_target, side="right"))
    if k >= len(fpr):
        return float(tpr[-1])
    f0, f1, t0, t1 = fpr[k - 1], fpr[k], tpr[k - 1], tpr[k]
    return float(t0 + (t1 - t0) * (fpr_target - f0) / (f1 - f0))


def auroc(pos, neg):
    """Probability that a positive scores above a negative, ties counted as one half."""
    neg_below = neg.sum() - np.cumsum(neg)
    return float(np.sum(pos * (neg_below + 0.5 * neg)) / (pos.sum() * neg.sum()))


def auprc(pos, neg):
    """Average precision, the step wise area under the precision recall curve."""
    tp = np.cumsum(pos)
    fp = np.cumsum(neg)
    keep = pos > 0
    precision = tp[keep] / (tp[keep] + fp[keep])
    return float(np.sum(precision * pos[keep]) / tp[-1])


def ranking_metrics(pos, neg):
    return {"sens_at_5fpr": sens_at_fpr(pos, neg), "auroc": auroc(pos, neg), "auprc": auprc(pos, neg)}


def threshold_at_specificity(scores, y, specificity):
    """Smallest threshold whose specificity on these rows is at least the target (alarm when score >= threshold)."""
    neg = np.sort(np.asarray(scores, dtype=float)[np.asarray(y) == 0])
    k = min(max(int(np.ceil(specificity * len(neg))), 1), len(neg))
    return float(np.nextafter(neg[k - 1], np.inf))


def operating_point(scores, y, patients, septic, threshold):
    """Row level sensitivity and false alarm rate, and patient level detection at a fixed threshold."""
    scores, y = np.asarray(scores), np.asarray(y)
    alarm = scores >= threshold
    out = {"sensitivity": float(alarm[y == 1].mean()), "false_alarm_rate": float(alarm[y == 0].mean())}
    septic = np.asarray(septic).astype(bool)
    pos_rows = (y == 1)
    pts_pos = np.unique(np.asarray(patients)[pos_rows])
    detected = np.unique(np.asarray(patients)[pos_rows & alarm])
    out["patients_detected"] = float(len(detected) / max(len(pts_pos), 1))
    non = ~septic
    out["false_alarms_per_100_patient_days"] = float(100.0 * alarm[non].sum() / max(non.sum() / 24.0, 1e-9))
    return out


def patient_bootstrap_weights(patients, n_resamples, seed):
    """Yield row weights for patient level bootstrap resamples."""
    codes, uniq_inv = np.unique(np.asarray(patients), return_inverse=True)
    rng = np.random.default_rng(seed)
    n = len(codes)
    for _ in range(n_resamples):
        counts = np.bincount(rng.integers(0, n, n), minlength=n)
        yield counts[uniq_inv.ravel()].astype(float)
