"""Kernel diagnostics: how concentrated the training Gram matrix is, and how well it aligns with the labels.

For a fidelity kernel on n qubits, kernel concentration means that off diagonal entries bunch up near one
value (for a very expressive map, near 2**-n), so the Gram matrix approaches the identity and carries little
information about the data. A bandwidth that is too large makes the map more expressive and the entries more
concentrated. These diagnostics are computed from the training Gram matrix only, never from test data.
"""

import numpy as np
import pandas as pd

from .classical import EXTENDED_GRIDS
from .config import HEADLINE_N
from .features import Preprocessor, feature_matrix
from .quantum import angles, kernel, statevectors
from .splits import balanced_draw


def gram_diagnostics(K, y):
    """Mean and variance of the off diagonal entries, and kernel target alignment with the plus or minus one labels."""
    n = len(K)
    off = K[np.triu_indices(n, k=1)]
    t = np.where(np.asarray(y) == 1, 1.0, -1.0)
    align = float(t @ K @ t / (n * np.linalg.norm(K)))
    H = np.eye(n) - 1.0 / n
    Kc = H @ K @ H
    tc = t - t.mean()
    centered = float(tc @ Kc @ tc / (np.linalg.norm(tc) ** 2 * np.linalg.norm(Kc))) if np.linalg.norm(Kc) > 0 else 0.0
    return {"mean_offdiag": float(off.mean()), "var_offdiag": float(off.var()), "alignment": align,
            "centered_alignment": centered}


def kernel_diagnostics(sets, out, n=HEADLINE_N, seed=0):
    """Training Gram matrix diagnostics at N rows of draw `seed`, for both maps over the extended bandwidths."""
    train = sets["train"]
    idx = balanced_draw(train, n, seed)
    X_raw = feature_matrix(train)[idx]
    pre = Preprocessor().fit(X_raw)
    X, y = pre.transform(X_raw), train["y"].to_numpy()[idx]
    rows = []
    for name, entangled in (("qsvm_zz", True), ("qsvm_product", False)):
        for reps in EXTENDED_GRIDS[name]["reps"]:
            for bandwidth in EXTENDED_GRIDS[name]["bandwidth"]:
                S = statevectors(angles(X, bandwidth), reps, entangled)
                K = kernel(S, S)
                rows.append({"model": name, "reps": reps, "bandwidth": bandwidth, "N": n, "seed": seed,
                             **gram_diagnostics(K, y)})
    pd.DataFrame(rows).to_csv(out / "kernel_diagnostics.csv", index=False)
