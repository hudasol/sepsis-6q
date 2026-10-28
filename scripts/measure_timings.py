"""Clean timing measurement for the matched models, in one process with nothing else running.

For every training size and draw seed it repeats the tuning, the final fit and the prediction on all evaluation
rows with the primary grids and the same rows as the primary run, and writes the timings together with the
hardware description to results/timings_clean.csv. The primary efficiency file is not touched.

Usage:
    python scripts/measure_timings.py [--out results/timings_clean.csv]
"""

import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import run_pipeline as rp  # noqa: E402
from sepsis6q import config as C  # noqa: E402
from sepsis6q.classical import MATCHED, fit_model, tune  # noqa: E402
from sepsis6q.features import Preprocessor, feature_matrix  # noqa: E402
from sepsis6q.splits import balanced_draw  # noqa: E402


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "timings_clean.csv"))
    args = ap.parse_args()
    args.limit = None
    with tempfile.TemporaryDirectory() as tmp:
        sets = rp.prepare(args, Path(tmp))
    train = sets["train"]
    X_raw, y_all, groups_all = feature_matrix(train), train["y"].to_numpy(), train["patient"].to_numpy()
    n_rows = sum(len(sets[s]) for s in rp.EVAL_SETS)
    hardware = {"cpu_model": cpu_model(), "logical_cores": os.cpu_count(), "platform": platform.platform(),
                "python": platform.python_version(), "numpy": np.__version__}
    rows = []
    for n in C.TRAIN_SIZES:
        for seed in C.DRAW_SEEDS:
            idx = balanced_draw(train, n, seed)
            pre = Preprocessor().fit(X_raw[idx])
            X, y, groups = pre.transform(X_raw[idx]), y_all[idx], groups_all[idx]
            for name in MATCHED:
                start = time.perf_counter()
                params, _ = tune(name, X, y, groups)
                tune_s = time.perf_counter() - start
                model, fit_s = fit_model(name, params, X, y)
                _, pred_s = rp.predict_all(model, pre, sets)
                if hasattr(model, "sv_states_"):
                    n_sv = len(model.sv_states_)
                else:
                    n_sv = int(model.support_.size) if hasattr(model, "support_") else -1
                rows.append({"model": name, "N": n, "seed": seed, "tune_seconds": tune_s, "train_seconds": fit_s,
                             "predict_seconds": pred_s, "predict_rows": n_rows, "n_support": n_sv, **hardware})
                rp.log(f"N={n} seed={seed} {name}: tune {tune_s:.1f}s fit {fit_s:.2f}s predict {pred_s:.1f}s")
                pd.DataFrame(rows).to_csv(args.out, index=False)
    rp.log("finished")


if __name__ == "__main__":
    main()
