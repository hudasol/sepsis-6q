"""Quantum resource counts, efficiency figures and the shot noise study."""

import json

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .config import SHOT_COUNTS
from .evaluate import ScoreIndex, ranking_metrics
from .features import Preprocessor, feature_matrix
from .quantum import N_QUBITS, QuantumKernelSVM, kernel, parameterized_feature_map
from .splits import balanced_draw

BASIS = ["rz", "sx", "x", "cx"]


def _counts(qc):
    ops = {k: v for k, v in qc.count_ops().items() if k not in ("barrier", "measure")}
    return {"depth": qc.depth(), "two_qubit_gates": int(ops.get("cx", 0)), "total_gates": int(sum(ops.values()))}


def circuit_resources(out):
    """Transpile one feature map circuit and one kernel (compute uncompute) circuit per map and repetition count."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    coupling = CouplingMap.from_line(N_QUBITS)
    rows = []
    for entangled in (True, False):
        for reps in (1, 2):
            fm = parameterized_feature_map(N_QUBITS, reps, entangled, "x")
            fm_y = parameterized_feature_map(N_QUBITS, reps, entangled, "y")
            kern = fm.compose(fm_y.inverse())
            kern.measure_all()
            for label, qc in (("feature map", fm), ("kernel circuit", kern)):
                t = transpile(qc, basis_gates=BASIS, coupling_map=coupling, optimization_level=1, seed_transpiler=0)
                rows.append({"map": "ZZ linear" if entangled else "product", "reps": reps, "circuit": label,
                             "qubits": N_QUBITS, **_counts(t)})
    pd.DataFrame(rows).to_csv(out / "resources.csv", index=False)


def efficiency_table(metrics, out):
    """Mean timings per model and N, and the number of distinct circuits each kernel estimator needs."""
    m = metrics[metrics["set"] == "val"]
    rows = []
    for (model, n), g in m.groupby(["model", "N"]):
        rows_eval = int(g["predict_rows"].iloc[0])
        row = {"model": model, "N": int(n), "tune_seconds": g["tune_seconds"].mean(),
               "train_seconds": g["train_seconds"].mean(), "predict_seconds": g["predict_seconds"].mean(),
               "predict_ms_per_1000_rows": 1000.0 * g["predict_seconds"].mean() / rows_eval * 1000.0,
               "eval_rows": rows_eval, "support_vectors": g["n_support"].mean()}
        if model.startswith("qsvm"):
            row["circuits_exact_states"] = int(n + rows_eval)
            row["circuits_sampled_pairwise"] = int(n * (n - 1) // 2 + rows_eval * n)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "efficiency.csv", index=False)


def _psd_project(K):
    vals, vecs = np.linalg.eigh(K)
    clipped = (vals < 0).sum()
    return (vecs * np.clip(vals, 0, None)) @ vecs.T, float(vals.min()), int(clipped)


def shot_noise_study(sets, out, metrics, shot_n, draw_seed=0, noise_seed=0):
    """Replace each kernel entry with a binomial estimate from a finite number of shots.

    The tuned configuration of the entangling QSVM at N = shot_n, draw seed 0, is
    reused. Training and test kernel entries are both sampled. A noisy training
    Gram matrix that is not positive semi definite is projected by clipping its
    negative eigenvalues to zero.
    """
    tuning = pd.read_csv(out / "tuning.csv")
    sel = tuning[(tuning["model"] == "qsvm_zz") & (tuning["N"] == shot_n) & (tuning["seed"] == draw_seed)]
    if sel.empty:
        return
    params = json.loads(sel["params"].iloc[0])
    train = sets["train"]
    idx = balanced_draw(train, shot_n, draw_seed)
    X_raw = feature_matrix(train)[idx]
    y = train["y"].to_numpy()[idx]
    pre = Preprocessor().fit(X_raw)
    model = QuantumKernelSVM(entangled=True, **params)
    S = model.states(pre.transform(X_raw))
    K = kernel(S, S)
    test = sets["test_A"]
    St = model.states(pre.transform(feature_matrix(test)))
    Kt = kernel(St, S)
    yt = test["y"].to_numpy()
    rng = np.random.default_rng(noise_seed)
    rows = []
    for shots in [None] + SHOT_COUNTS:
        if shots is None:
            Kn, Ktn, min_eig, clipped = K, Kt, float(np.linalg.eigvalsh(K).min()), 0
        else:
            upper = rng.binomial(shots, np.clip(np.triu(K, 1), 0, 1)) / shots
            Kn = upper + upper.T + np.eye(len(K))
            Kn, min_eig, clipped = _psd_project(Kn)
            Ktn = rng.binomial(shots, np.clip(Kt, 0, 1)) / shots
        svc = SVC(kernel="precomputed", C=params["C"]).fit(Kn, y)
        scores = svc.decision_function(Ktn)
        res = ranking_metrics(*ScoreIndex(scores, yt).tallies())
        rows.append({"shots": "exact" if shots is None else shots, "N": shot_n, **res,
                     "min_eigenvalue_before_projection": min_eig, "negative_eigenvalues_clipped": clipped})
    pd.DataFrame(rows).to_csv(out / "shot_noise.csv", index=False)
