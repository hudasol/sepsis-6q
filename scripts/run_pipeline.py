"""Run the SEPSIS-6Q benchmark end to end and write small result files to results/.

Usage:
    python scripts/run_pipeline.py            # full run on all downloaded files
    python scripts/run_pipeline.py --smoke    # about 300 patients per set, minutes
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sepsis6q import config as C  # noqa: E402
from sepsis6q.classical import CLASSICAL, MATCHED, fit_model, make_classical, scores_of, tune  # noqa: E402
from sepsis6q.data import label_behaviour, load_set  # noqa: E402
from sepsis6q.evaluate import (ScoreIndex, operating_point, patient_bootstrap_weights,  # noqa: E402
                               ranking_metrics, threshold_at_specificity)
from sepsis6q.features import Preprocessor, add_features, early_warning_rows, feature_matrix  # noqa: E402
from sepsis6q.splits import balanced_draw, split_patients, stratified_subset  # noqa: E402

EVAL_SETS = ["val", "test_A", "test_B"]
TEST_SETS = ["test_A", "test_B"]
REFERENCE = ["ref_logreg_full", "ref_hgb_full"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=float))


def describe(rows):
    pts = rows.groupby("patient")["septic"].max()
    return {"patients": int(len(pts)), "septic_patients": int(pts.sum()), "rows": int(len(rows)),
            "positive_rows": int(rows["y"].sum()), "row_prevalence": float(rows["y"].mean())}


def prepare(args, out):
    """Load both sets, build the early warning rows, split set A and draw the fixed evaluation subsets."""
    rows, counts = {}, {}
    for s in ("A", "B"):
        raw = load_set(s, limit=args.limit)
        log(f"set {s}: {raw['patient'].nunique()} patients, {len(raw)} raw rows")
        counts[f"raw_{s}"] = {"files": int(raw["patient"].nunique()), "rows": int(len(raw))}
        counts[f"labels_{s}"] = label_behaviour(raw)
        rows[s] = early_warning_rows(add_features(raw))
        counts[f"task_{s}"] = describe(rows[s])
    split = split_patients(rows["A"])
    seed = C.SPLIT_SEED + 1

    def take(frame, patients):
        return frame[frame["patient"].isin(patients)].reset_index(drop=True)

    sets = {
        "train": take(rows["A"], split["train"]),
        "val": take(rows["A"], stratified_subset(rows["A"], split["val"], C.VAL_SEPTIC_PATIENTS, seed)),
        "test_A": take(rows["A"], stratified_subset(rows["A"], split["test"], C.TEST_SEPTIC_PATIENTS, seed)),
        "test_B": take(rows["B"], stratified_subset(rows["B"], rows["B"]["patient"].unique(),
                                                    C.TEST_SEPTIC_PATIENTS, seed)),
    }
    counts["split_A_patients"] = {k: int(len(v)) for k, v in split.items()}
    for k, v in sets.items():
        counts[f"eval_{k}"] = describe(v)
    save_json(counts, out / "data_counts.json")
    return sets


def evaluate_scores(name, scores, sets, extra):
    """Ranking metrics on every set and deployment style operating points from validation thresholds."""
    rows = []
    val = sets["val"]
    thresholds = {spec: threshold_at_specificity(scores["val"], val["y"].to_numpy(), spec)
                  for spec in C.VAL_SPECIFICITIES}
    for set_name in EVAL_SETS:
        d = sets[set_name]
        y = d["y"].to_numpy()
        pos, neg = ScoreIndex(scores[set_name], y).tallies()
        row = {**extra, "model": name, "set": set_name, **ranking_metrics(pos, neg), "prevalence": float(y.mean())}
        for spec, thr in thresholds.items():
            op = operating_point(scores[set_name], y, d["patient"].to_numpy(), d["septic"].to_numpy(), thr)
            tag = f"spec{int(round(spec * 100))}"
            for k, v in op.items():
                row[f"{tag}_{k}"] = v
        rows.append(row)
    return rows


def predict_all(model, pre, sets):
    out, total = {}, 0.0
    for s in EVAL_SETS:
        X = pre.transform(feature_matrix(sets[s]))
        start = time.perf_counter()
        out[s] = scores_of(model, X)
        total += time.perf_counter() - start
    return out, total


def reference_models(sets, out, score_dir):
    """Context only: models trained on the whole training pool with the same eight features."""
    train = sets["train"]
    pre = Preprocessor().fit(feature_matrix(train))
    X, y = pre.transform(feature_matrix(train)), train["y"].to_numpy()
    refs = {
        "ref_logreg_full": make_classical("logreg", {"C": 1.0}).set_params(class_weight="balanced"),
        "ref_hgb_full": make_classical("hgb", {"learning_rate": 0.1, "max_depth": 4, "max_iter": 300}),
    }
    metrics = []
    for name, model in refs.items():
        start = time.perf_counter()
        model.fit(X, y)
        fit_time = time.perf_counter() - start
        scores, pred_time = predict_all(model, pre, sets)
        np.savez_compressed(score_dir / f"{name}.npz", **scores)
        metrics += evaluate_scores(name, scores, sets, {"N": len(train), "seed": -1, "train_seconds": fit_time,
                                                        "predict_seconds": pred_time})
        log(f"{name}: done")
    pd.DataFrame(metrics).to_csv(out / "reference_metrics.csv", index=False)


def matched_runs(sets, sizes, seeds, out, score_dir):
    """Every matched model on the same balanced draws, tuned by patient grouped cross validation."""
    train = sets["train"]
    X_raw, y_all, groups_all = feature_matrix(train), train["y"].to_numpy(), train["patient"].to_numpy()
    metrics_path, tuning_path = out / "metrics_runs.csv", out / "tuning.csv"
    metrics = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    tuning = pd.read_csv(tuning_path).to_dict("records") if tuning_path.exists() else []
    done = {(r["N"], r["seed"], r["model"]) for r in tuning}
    n_rows = sum(len(sets[s]) for s in EVAL_SETS)
    for n in sizes:
        for seed in seeds:
            idx = balanced_draw(train, n, seed)
            pre = Preprocessor().fit(X_raw[idx])
            X, y, groups = pre.transform(X_raw[idx]), y_all[idx], groups_all[idx]
            for name in MATCHED:
                if (n, seed, name) in done:
                    continue
                start = time.perf_counter()
                params, table = tune(name, X, y, groups)
                tune_time = time.perf_counter() - start
                model, fit_time = fit_model(name, params, X, y)
                scores, pred_time = predict_all(model, pre, sets)
                np.savez_compressed(score_dir / f"{name}_N{n}_s{seed}.npz", **scores)
                if hasattr(model, "sv_states_"):
                    n_sv = len(model.sv_states_)
                elif hasattr(model, "support_"):
                    n_sv = int(model.support_.size)
                else:
                    n_sv = -1
                extra = {"N": n, "seed": seed, "tune_seconds": tune_time, "train_seconds": fit_time,
                         "predict_seconds": pred_time, "predict_rows": n_rows, "n_support": n_sv}
                metrics += evaluate_scores(name, scores, sets, extra)
                tuning.append({"N": n, "seed": seed, "model": name, "configs": len(table),
                               "cv_auroc": max(r["cv_auroc"] for r in table), "params": json.dumps(params)})
                pd.DataFrame(metrics).to_csv(metrics_path, index=False)
                pd.DataFrame(tuning).to_csv(tuning_path, index=False)
                log(f"N={n} seed={seed} {name}: {params} tune {tune_time:.1f}s fit {fit_time:.1f}s "
                    f"predict {pred_time:.1f}s")
    return pd.DataFrame(metrics)


def bootstrap(sets, metrics, headline_n, seeds, n_resamples, out, score_dir):
    """Patient level bootstrap at the headline N. Each metric is the mean over draw seeds within a resample."""
    val = metrics[(metrics["N"] == headline_n) & (metrics["set"] == "val")]
    best_classical = val[val["model"].isin(CLASSICAL)].groupby("model")["sens_at_5fpr"].mean().idxmax()
    models = MATCHED + REFERENCE
    keys = ("sens_at_5fpr", "auroc", "auprc")
    rows = []
    for set_name in TEST_SETS:
        d = sets[set_name]
        y = d["y"].to_numpy()
        index = {}
        for m in models:
            files = [score_dir / f"{m}.npz"] if m in REFERENCE else \
                [score_dir / f"{m}_N{headline_n}_s{s}.npz" for s in seeds]
            index[m] = [ScoreIndex(np.load(f)[set_name], y) for f in files]

        def mean_metrics(m, w=None):
            vals = [ranking_metrics(*ix.tallies(w)) for ix in index[m]]
            return {k: float(np.mean([v[k] for v in vals])) for k in keys}

        point = {m: mean_metrics(m) for m in models}
        draws = {m: {k: [] for k in keys} for m in models}
        for w in patient_bootstrap_weights(d["patient"].to_numpy(), n_resamples, seed=C.SPLIT_SEED):
            for m in models:
                for k, v in mean_metrics(m, w).items():
                    draws[m][k].append(v)
        for m in models:
            for k in keys:
                arr = np.array(draws[m][k])
                rows.append({"set": set_name, "comparison": m, "metric": k, "estimate": point[m][k],
                             "ci_low": float(np.nanpercentile(arr, 2.5)), "ci_high": float(np.nanpercentile(arr, 97.5))})
        for a, b in (("qsvm_zz", best_classical), ("qsvm_zz", "qsvm_product")):
            for k in keys:
                diff = np.array(draws[a][k]) - np.array(draws[b][k])
                rows.append({"set": set_name, "comparison": f"{a} minus {b}", "metric": k,
                             "estimate": point[a][k] - point[b][k],
                             "ci_low": float(np.nanpercentile(diff, 2.5)), "ci_high": float(np.nanpercentile(diff, 97.5))})
        log(f"bootstrap {set_name}: done")
    pd.DataFrame(rows).to_csv(out / "bootstrap.csv", index=False)
    save_json({"headline_N": headline_n, "best_matched_classical": best_classical, "resamples": n_resamples,
               "selection": "highest mean validation sens_at_5fpr over draws"}, out / "headline.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="patient files per set")
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=None, help="results directory")
    ap.add_argument("--skip-shots", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="reuse saved scores, redo bootstrap and reports")
    args = ap.parse_args()
    if args.smoke:
        args.limit = args.limit or 300
        sizes, seeds, n_boot, headline = [50, 100], [0, 1], 50, 100
        out = C.RESULTS_DIR / "smoke"
    else:
        sizes, seeds, n_boot, headline = C.TRAIN_SIZES, C.DRAW_SEEDS, C.BOOTSTRAP_RESAMPLES, C.HEADLINE_N
        out = C.RESULTS_DIR
    sizes, seeds = args.sizes or sizes, args.seeds or seeds
    out = Path(args.out).resolve() if args.out else out
    out.mkdir(parents=True, exist_ok=True)
    score_dir = C.CACHE_DIR / ("scores_" + out.name)
    score_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    sets = prepare(args, out)
    log("data prepared")
    if args.report_only:
        metrics = pd.read_csv(out / "metrics_runs.csv")
    else:
        if not (score_dir / "ref_hgb_full.npz").exists():
            reference_models(sets, out, score_dir)
        metrics = matched_runs(sets, sizes, seeds, out, score_dir)
    if headline in set(metrics["N"]):
        bootstrap(sets, metrics, headline, seeds, n_boot, out, score_dir)

    from sepsis6q import resources
    resources.circuit_resources(out)
    resources.efficiency_table(metrics, out)
    if not args.skip_shots:
        resources.shot_noise_study(sets, out, metrics, shot_n=50 if args.smoke else C.SHOT_N)
    from sepsis6q import plots
    plots.make_all(out)
    save_json({"total_seconds": time.perf_counter() - t0, "smoke": args.smoke, "sizes": sizes, "seeds": seeds,
               "bootstrap_resamples": n_boot, "patient_file_limit": args.limit}, out / "run_info.json")
    log("finished")


if __name__ == "__main__":
    main()
