"""The sensitivity grids must widen the primary grids without changing them."""

import numpy as np

from sepsis6q.classical import EXTENDED_GRIDS, GRIDS, grid
from sepsis6q.diagnostics import gram_diagnostics


def test_primary_grids_are_unchanged():
    assert GRIDS["rbf_svm"]["gamma"] == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert GRIDS["qsvm_zz"]["bandwidth"] == [0.25, 0.5, 1.0]
    assert GRIDS["qsvm_product"]["bandwidth"] == [0.25, 0.5, 1.0]


def test_extended_grids_contain_every_primary_value():
    for name, primary in GRIDS.items():
        for key, values in primary.items():
            assert set(values) <= set(EXTENDED_GRIDS[name][key]), (name, key)
    assert min(EXTENDED_GRIDS["qsvm_zz"]["bandwidth"]) < min(GRIDS["qsvm_zz"]["bandwidth"])
    assert min(EXTENDED_GRIDS["rbf_svm"]["gamma"]) < min(GRIDS["rbf_svm"]["gamma"])
    assert len(grid("hgb", EXTENDED_GRIDS)) == len(grid("hgb"))


def test_gram_diagnostics_on_known_matrices():
    y = np.array([1, 1, 0, 0])
    identity = gram_diagnostics(np.eye(4), y)
    assert identity["mean_offdiag"] == 0.0 and identity["var_offdiag"] == 0.0
    assert abs(identity["alignment"] - 0.5) < 1e-12
    t = np.array([1.0, 1.0, -1.0, -1.0])
    perfect = gram_diagnostics(np.outer(t, t), y)
    assert abs(perfect["alignment"] - 1.0) < 1e-12
    assert abs(perfect["centered_alignment"] - 1.0) < 1e-12
    ones = gram_diagnostics(np.ones((4, 4)), y)
    assert ones["var_offdiag"] == 0.0 and ones["mean_offdiag"] == 1.0
