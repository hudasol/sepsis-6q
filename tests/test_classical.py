import numpy as np
from sklearn.svm import SVC


def test_fast_rbf_decision_matches_sklearn():
    from sepsis6q.classical import RBFSVM

    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(300, 8))
    y = (X[:, 0] * X[:, 2] > 0.25).astype(int)
    Xt = rng.uniform(0, 1, size=(1000, 8))
    fast = RBFSVM(C=1.0, gamma=2.0, chunk=97).fit(X, y)
    ref = SVC(kernel="rbf", C=1.0, gamma=2.0).fit(X, y)
    assert np.max(np.abs(fast.decision_function(Xt) - ref.decision_function(Xt))) < 1e-8
