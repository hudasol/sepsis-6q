import numpy as np
import pytest
from sklearn.svm import SVC

from sepsis6q.quantum import QuantumKernelSVM, kernel, qiskit_statevector, statevectors


@pytest.mark.parametrize("entangled", [True, False])
@pytest.mark.parametrize("reps", [1, 2, 3])
def test_numpy_states_equal_qiskit(entangled, reps):
    rng = np.random.default_rng(reps + 10 * entangled)
    x = rng.uniform(0, np.pi, size=(6, 8))
    ours = statevectors(x, reps, entangled)
    ref = np.array([qiskit_statevector(v, reps, entangled) for v in x])
    assert np.max(np.abs(ours - ref)) < 1e-9


@pytest.mark.parametrize("reps", [1, 2])
def test_kernel_matches_qiskit_library_zz_feature_map(reps):
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(7)
    x = rng.uniform(0, np.pi, size=(4, 8))
    fm = zz_feature_map(8, reps=reps, entanglement="linear")
    lib = np.array([Statevector(fm.assign_parameters(v)).data for v in x])
    ours = statevectors(x, reps, True)
    assert np.max(np.abs(kernel(lib, lib) - kernel(ours, ours))) < 1e-9


def test_gram_is_valid_kernel():
    x = np.random.default_rng(0).uniform(0, 1, size=(40, 8))
    S = statevectors(np.pi * x, 2, True)
    K = kernel(S, S)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_chunked_decision_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(120, 8))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    Xt = rng.uniform(0, 1, size=(50, 8))
    model = QuantumKernelSVM(C=1.0, bandwidth=0.5, reps=2, chunk=7).fit(X, y)
    S, St = model.states(X), model.states(Xt)
    ref = SVC(kernel="precomputed", C=1.0).fit(kernel(S, S), y).decision_function(kernel(St, S))
    assert np.max(np.abs(model.decision_function(Xt) - ref)) < 1e-6

