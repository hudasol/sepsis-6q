"""Quantum kernel: Qiskit reference feature maps and a batched numpy statevector simulator.

The entangling map follows the ZZFeatureMap construction with linear entanglement.
Each repetition is a Hadamard layer on every qubit followed by a diagonal phase:
P(2 x_i) on qubit i, and for each neighbouring pair (i, i+1) the block
CX(i, i+1) P(2 (pi - x_i)(pi - x_{i+1})) CX(i, i+1). The product map control keeps
the Hadamard layer and the single qubit phases and drops the pairwise terms.

Inputs are features scaled to [0, 1]; the angles are x = bandwidth * pi * scaled.
"""

import numpy as np
from sklearn.svm import SVC

N_QUBITS = 8


def linear_pairs(n=N_QUBITS):
    return [(i, i + 1) for i in range(n - 1)]


def angles(X_scaled, bandwidth):
    return np.pi * bandwidth * np.asarray(X_scaled, dtype=float)


# Qiskit reference definition


def feature_map_circuit(x, reps, entangled):
    """Bound Qiskit circuit for one input vector of angles."""
    from qiskit import QuantumCircuit

    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.p(2.0 * x[i], i)
        if entangled:
            for i, j in linear_pairs(n):
                qc.cx(i, j)
                qc.p(2.0 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def parameterized_feature_map(n, reps, entangled, prefix="x"):
    """Same map with free parameters, used for resource counts."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = ParameterVector(prefix, n)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.p(2.0 * x[i], i)
        if entangled:
            for i, j in linear_pairs(n):
                qc.cx(i, j)
                qc.p(2.0 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def qiskit_statevector(x, reps, entangled):
    from qiskit.quantum_info import Statevector

    return Statevector(feature_map_circuit(x, reps, entangled)).data


# Batched numpy simulator


def _bits(n):
    k = np.arange(2 ** n)
    return ((k[:, None] >> np.arange(n)[None, :]) & 1).astype(float)


def _hadamard(n):
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    out = np.ones((1, 1))
    for _ in range(n):
        out = np.kron(H, out)
    return out


def _phase_matrix(n, entangled):
    """Coefficients M such that the phase angle of one repetition is F(x) @ M.

    F(x) holds x_i for every qubit and, for the entangling map, the pair terms
    (pi - x_i)(pi - x_j). Basis state k has bit i equal to (k >> i) & 1, which
    matches the qubit ordering of Qiskit.
    """
    Z = _bits(n)
    rows = [2.0 * Z[:, i] for i in range(n)]
    if entangled:
        rows += [2.0 * np.abs(Z[:, i] - Z[:, j]) for i, j in linear_pairs(n)]
    return np.array(rows)


def _phase_inputs(x, entangled):
    cols = [x]
    if entangled:
        cols.append(np.stack([(np.pi - x[:, i]) * (np.pi - x[:, j]) for i, j in linear_pairs(x.shape[1])], axis=1))
    return np.concatenate(cols, axis=1)


def phases(x, entangled):
    """Diagonal phase of one repetition for a batch of angle vectors, shape (B, 2**n)."""
    theta = _phase_inputs(x, entangled) @ _phase_matrix(x.shape[1], entangled)
    return np.exp(1j * theta)


def statevectors(x, reps, entangled):
    """Statevectors of the feature map for a batch of angle vectors.

    The Hadamard layer is applied as one product with the dense Walsh Hadamard
    matrix, and the diagonal phase as an elementwise product.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = x.shape[1]
    diag = phases(x, entangled)
    psi = diag / np.sqrt(2 ** n)
    if reps > 1:
        H = _hadamard(n)
        for _ in range(reps - 1):
            psi = (psi @ H) * diag
    return psi


def kernel(S_a, S_b):
    """Fidelity kernel |<psi(a)|psi(b)>|^2 between two batches of states."""
    return np.abs(S_a.conj() @ S_b.T) ** 2


class QuantumKernelSVM:
    """SVM on the fidelity kernel of the feature map, with exact statevector simulation."""

    def __init__(self, C=1.0, bandwidth=1.0, reps=1, entangled=True, chunk=4096):
        self.C, self.bandwidth, self.reps, self.entangled, self.chunk = C, bandwidth, reps, entangled, chunk

    def states(self, X_scaled):
        return statevectors(angles(X_scaled, self.bandwidth), self.reps, self.entangled)

    def fit(self, X_scaled, y, gram=None, states=None):
        S = self.states(X_scaled) if states is None else states
        K = kernel(S, S) if gram is None else gram
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(K, y)
        sv = self.svc_.support_
        self.sv_states_ = S[sv]
        self.dual_coef_ = self.svc_.dual_coef_[0]
        self.intercept_ = float(self.svc_.intercept_[0])
        return self

    def decision_function(self, X_scaled):
        """Decision values in chunks, using support vectors only.

        Overlaps for prediction are computed in single precision complex
        arithmetic, which changes kernel values by about 1e-7 and halves the time.
        """
        X_scaled = np.asarray(X_scaled)
        out = np.empty(len(X_scaled))
        sv = self.sv_states_.astype(np.complex64)
        for start in range(0, len(X_scaled), self.chunk):
            S = self.states(X_scaled[start:start + self.chunk]).astype(np.complex64)
            out[start:start + self.chunk] = kernel(S, sv).astype(float) @ self.dual_coef_ + self.intercept_
        return out
