"""Classical ceiling reference: boosted trees on every raw column, not matched and not usable with eight qubits.

The matched comparison restricts every model to the same eight features. This script shows what a strong classical
model reaches when it is not limited in that way: gradient boosted trees trained on the whole training pool of set A
with all 34 clinical columns (forward filled within each patient, past values only), the 5 demographic and unit
columns, ICULOS, the two trend features, and one missingness indicator per clinical column (1 when the value was
measured in that hour). It uses the same rows, patient split, evaluation subsets, operating points and bootstrap as
the matched comparison, and the same boosted tree settings as the full pool reference with eight features.

Usage:
    python scripts/ceiling_reference.py
"""

import argparse
import json
import pickle
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
from sepsis6q.classical import make_classical, scores_of  # noqa: E402
from sepsis6q.data import list_patient_files  # noqa: E402
from sepsis6q.evaluate import ScoreIndex, patient_bootstrap_weights, ranking_metrics  # noqa: E402

CLINICAL = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
            "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
            "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT",
            "WBC", "Fibrinogen", "Platelets"]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
INDICATORS = [f"m_{c}" for c in CLINICAL]
LABS = [c for c in CLINICAL if c not in C.VITALS]
FEATURES = CLINICAL + ["HR_d6", "Resp_d6"] + DEMOGRAPHICS + ["ICULOS"] + INDICATORS
MODEL_NAME = "ceiling_hgb_all_columns"


def load_full(set_name, limit=None, use_cache=True):
    """Like sepsis6q.data.load_set, but with every column and one measured this hour indicator per clinical column."""
    files = list_patient_files(set_name, limit)
    cache = C.CACHE_DIR / f"full_set{set_name}_{len(files)}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    frames = []
    for path in files:
        df = pd.read_csv(path, sep="|")
        df.insert(0, "patient", f"{set_name}_{Path(path).stem}")
        for col in CLINICAL:
            df[f"m_{col}"] = df[col].notna().astype(np.int8)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["SepsisLabel"] = out["SepsisLabel"].astype(np.int8)
    if use_cache:
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out


def design(frame):
    """Feature matrix: laboratory columns are carried forward within each patient (past values only)."""
    f = frame[["patient"] + CLINICAL + ["HR_d6", "Resp_d6"] + DEMOGRAPHICS + ["ICULOS"] + INDICATORS].copy()
    f[LABS] = f.groupby("patient", sort=False)[LABS].ffill()
    return f[FEATURES].to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    args = ap.parse_args()
    args.limit = None
    out = Path(args.out)
    rp.load_set = load_full  # the shared preparation then keeps every column
    with tempfile.TemporaryDirectory() as tmp:
        sets = rp.prepare(args, Path(tmp))
    train = sets["train"]
    X, y = design(train), train["y"].to_numpy()
    # Columns with fewer than two distinct observed values in the training pool carry no information, and the
    # scikit-learn histogram binning cannot handle them, so they are dropped and listed.
    keep = [j for j in range(X.shape[1]) if len(np.unique(X[~np.isnan(X[:, j]), j])) > 1]
    dropped = [FEATURES[j] for j in range(X.shape[1]) if j not in keep]
    X = X[:, keep]
    rp.log(f"training pool: {X.shape[0]} rows, {X.shape[1]} features, dropped {dropped}")
    model = make_classical("hgb", {"learning_rate": 0.1, "max_depth": 4, "max_iter": 300})
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start
    rp.log(f"fit {fit_seconds:.1f}s")
    scores = {s: scores_of(model, design(sets[s])[:, keep]) for s in rp.EVAL_SETS}
    rows = rp.evaluate_scores(MODEL_NAME, scores, sets, {"N": len(train), "seed": -1, "train_seconds": fit_seconds,
                                                         "predict_seconds": float("nan")})
    pd.DataFrame(rows).to_csv(out / "ceiling_metrics.csv", index=False)

    boot = []
    for set_name in rp.TEST_SETS:
        d = sets[set_name]
        index = ScoreIndex(scores[set_name], d["y"].to_numpy())
        point = ranking_metrics(*index.tallies())
        draws = {k: [] for k in ("sens_at_5fpr", "auroc", "auprc")}
        for w in patient_bootstrap_weights(d["patient"].to_numpy(), C.BOOTSTRAP_RESAMPLES, seed=C.SPLIT_SEED):
            m = ranking_metrics(*index.tallies(w))
            for k in draws:
                draws[k].append(m[k])
        for k, v in draws.items():
            arr = np.array(v)
            boot.append({"set": set_name, "metric": k, "estimate": point[k],
                         "ci_low": float(np.nanpercentile(arr, 2.5)), "ci_high": float(np.nanpercentile(arr, 97.5))})
    pd.DataFrame(boot).to_csv(out / "ceiling_bootstrap.csv", index=False)
    (out / "ceiling_info.json").write_text(json.dumps({
        "model": MODEL_NAME, "features": len(keep), "training_rows": int(len(train)), "dropped_columns": dropped,
        "feature_groups": {"clinical columns": len(CLINICAL), "trend features": 2, "demographic and unit columns": 5,
                           "ICULOS": 1, "missingness indicators": len(INDICATORS)},
        "bootstrap_resamples": C.BOOTSTRAP_RESAMPLES, "fit_seconds": fit_seconds}, indent=2))
    rp.log("finished")


if __name__ == "__main__":
    main()
