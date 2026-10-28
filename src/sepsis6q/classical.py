"""Matched model grids and the patient grouped cross validation used to tune every model."""

import itertools
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

from .config import CV_FOLDS
from .quantum import QuantumKernelSVM, angles, kernel, statevectors

GRIDS = {
    "logreg": {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "rbf_svm": {"C": [0.1, 1.0, 10.0], "gamma": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]},
    "hgb": {"learning_rate": [0.03, 0.1, 0.3], "max_depth": [2, 3, 4], "max_iter": [100, 300]},
    "qsvm_zz": {"C": [0.1, 1.0, 10.0], "bandwidth": [0.25, 0.5, 1.0], "reps": [1, 2]},
    "qsvm_product": {"C": [0.1, 1.0, 10.0], "bandwidth": [0.25, 0.5, 1.0], "reps": [1, 2]},
}
MATCHED = list(GRIDS)
CLASSICAL = ["logreg", "rbf_svm", "hgb"]
QUANTUM = ["qsvm_zz", "qsvm_product"]

# Wider grids for the post hoc sensitivity analysis (README section 7.8). The primary GRIDS above are not
# changed. The RBF gamma range and the QSVM bandwidth range are both widened towards smaller values, and every
# primary value stays inside the wider grid, so the sensitivity analysis can only differ from the primary
# analysis where a new value wins the same cross validation. Logistic regression and boosted trees keep their
# primary grids.
EXTENDED_GRIDS = {
    **GRIDS,
    "rbf_svm": {"C": [0.1, 1.0, 10.0], "gamma": [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]},
    "qsvm_zz": {"C": [0.1, 1.0, 10.0], "bandwidth": [0.02, 0.04, 0.08, 0.16, 0.25, 0.5, 1.0], "reps": [1, 2]},
    "qsvm_product": {"C": [0.1, 1.0, 10.0], "bandwidth": [0.02, 0.04, 0.08, 0.16, 0.25, 0.5, 1.0],
                     "reps": [1, 2]},
}
# The primary run also hit grid edges for logistic regression (C) and boosted trees (learning rate, depth), so a
# second sensitivity run widens every model, to check that widening only the kernel models is not what matters.
EXTENDED_ALL_GRIDS = {
    **EXTENDED_GRIDS,
    "logreg": {"C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "hgb": {"learning_rate": [0.01, 0.03, 0.1, 0.3], "max_depth": [1, 2, 3, 4], "max_iter": [100, 300, 600]},
}
GRID_SETS = {"primary": GRIDS, "extended": EXTENDED_GRIDS, "extended_all": EXTENDED_ALL_GRIDS}


def grid(name, grids=None):
    grids = GRIDS if grids is None else grids
    keys = list(grids[name])
    return [dict(zip(keys, vals)) for vals in itertools.product(*grids[name].values())]


class RBFSVM(SVC):
    """scikit-learn RBF SVC whose decision function is evaluated in chunks with numpy.

    Fitting is unchanged. Prediction uses the fitted support vectors, dual
    coefficients and intercept, which gives the same values as libsvm but runs
    multithreaded through matrix products.
    """

    def __init__(self, C=1.0, gamma=1.0, chunk=8192):
        super().__init__(kernel="rbf", C=C, gamma=gamma)
        self.chunk = chunk

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        sv = self.support_vectors_
        sv_sq = (sv ** 2).sum(axis=1)
        out = np.empty(len(X))
        for start in range(0, len(X), self.chunk):
            x = X[start:start + self.chunk]
            d2 = (x ** 2).sum(axis=1)[:, None] + sv_sq[None, :] - 2.0 * x @ sv.T
            out[start:start + self.chunk] = np.exp(-self.gamma * np.maximum(d2, 0.0)) @ self.dual_coef_[0]
        return out + self.intercept_[0]


def make_classical(name, params):
    if name == "logreg":
        return LogisticRegression(max_iter=5000, **params)
    if name == "rbf_svm":
        return RBFSVM(**params)
    if name == "hgb":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=0, **params)
    raise KeyError(name)


def scores_of(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def cv_splits(groups):
    return list(GroupKFold(n_splits=CV_FOLDS).split(np.zeros(len(groups)), groups=groups))


def tune(name, X, y, groups, grids=None):
    """Patient grouped 3 fold cross validation, AUROC on the held out fold. Returns best params and the table.

    `grids` defaults to the primary GRIDS; the sensitivity analysis passes EXTENDED_GRIDS."""
    folds = cv_splits(groups)
    table = []
    if name in ("qsvm_zz", "qsvm_product"):
        entangled = name == "qsvm_zz"
        grams = {}
        for params in grid(name, grids):
            key = (params["bandwidth"], params["reps"])
            if key not in grams:
                S = statevectors(angles(X, params["bandwidth"]), params["reps"], entangled)
                grams[key] = kernel(S, S)
            K = grams[key]
            aucs = []
            for tr, te in folds:
                svc = SVC(kernel="precomputed", C=params["C"]).fit(K[np.ix_(tr, tr)], y[tr])
                aucs.append(roc_auc_score(y[te], svc.decision_function(K[np.ix_(te, tr)])))
            table.append({**params, "cv_auroc": float(np.mean(aucs))})
    else:
        for params in grid(name, grids):
            aucs = []
            for tr, te in folds:
                model = make_classical(name, params).fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], scores_of(model, X[te])))
            table.append({**params, "cv_auroc": float(np.mean(aucs))})
    best = max(table, key=lambda r: r["cv_auroc"])
    params = {k: v for k, v in best.items() if k != "cv_auroc"}
    return params, table


def fit_model(name, params, X, y):
    """Fit one model on all N rows and return it with its training time in seconds."""
    start = time.perf_counter()
    if name in ("qsvm_zz", "qsvm_product"):
        model = QuantumKernelSVM(entangled=name == "qsvm_zz", **params).fit(X, y)
    else:
        model = make_classical(name, params).fit(X, y)
    return model, time.perf_counter() - start
