"""Render every number in README.md from the files in results/.

Each generated block sits between <!-- BEGIN name --> and <!-- END name --> markers.
Figures are regenerated from the same files.

Usage:
    python scripts/render_readme_tables.py [--results results]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sepsis6q.classical import GRIDS  # noqa: E402

NAMES = {
    "logreg": "Logistic regression",
    "rbf_svm": "RBF SVM",
    "hgb": "Gradient boosted trees",
    "qsvm_zz": "QSVM, ZZ entangling map",
    "qsvm_product": "QSVM, product map (control)",
    "ref_logreg_full": "Reference: logistic regression, full pool",
    "ref_hgb_full": "Reference: boosted trees, full pool",
}
SETS = {"test_A": "Internal test (set A)", "test_B": "External test (set B)"}
MATCHED = ["logreg", "rbf_svm", "hgb", "qsvm_zz", "qsvm_product"]


def f3(x):
    return f"{x:.3f}"


def ci(r):
    return f"{r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def load(res):
    def read(name):
        p = res / name
        if not p.exists():
            return None
        return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)

    return {k: read(v) for k, v in {
        "counts": "data_counts.json", "metrics": "metrics_runs.csv", "ref": "reference_metrics.csv",
        "boot": "bootstrap.csv", "headline": "headline.json", "res": "resources.csv", "eff": "efficiency.csv",
        "shots": "shot_noise.csv", "tuning": "tuning.csv", "info": "run_info.json"}.items()}


def verdict(row):
    if row["ci_low"] > 0:
        return "higher, and the 95% interval excludes zero"
    if row["ci_high"] < 0:
        return "lower, and the 95% interval excludes zero"
    return "not different at this sample size (the 95% interval includes zero)"


def block_summary(d):
    b, h = d["boot"], d["headline"]
    best = h["best_matched_classical"]
    parts = []
    for s, label in SETS.items():
        q = b[(b["set"] == s) & (b["comparison"] == "qsvm_zz") & (b["metric"] == "sens_at_5fpr")].iloc[0]
        c = b[(b["set"] == s) & (b["comparison"] == best) & (b["metric"] == "sens_at_5fpr")].iloc[0]
        diff = b[(b["set"] == s) & (b["comparison"] == f"qsvm_zz minus {best}")
                 & (b["metric"] == "sens_at_5fpr")].iloc[0]
        parts.append(
            f"On the {label[0].lower() + label[1:]}, the entangling QSVM reached a sensitivity of {ci(q)} at a 5% false positive "
            f"rate, against {ci(c)} for the best matched classical model ({NAMES[best]}, chosen on validation "
            f"data). The paired difference was {ci(diff)}, so the QSVM was {verdict(diff)}.")
    ctrl = b[(b["set"] == "test_A") & (b["comparison"] == "qsvm_zz minus qsvm_product")
             & (b["metric"] == "sens_at_5fpr")].iloc[0]
    parts.append(f"Against its own product map control on the internal test set, the entangling map was "
                 f"{verdict(ctrl)} (difference {ci(ctrl)}).")
    return f"Headline result at N = {h['headline_N']} balanced training rows, mean over 3 draws, " \
           f"patient level bootstrap with {h['resamples']} resamples. " + " ".join(parts)


def block_data(d):
    c = d["counts"]
    rows = []
    for s in ("A", "B"):
        raw, lab, task = c[f"raw_{s}"], c[f"labels_{s}"], c[f"task_{s}"]
        rows.append([f"Set {s}", raw["files"], raw["rows"], lab["septic_patients"],
                     f"{lab['monotone_septic']} of {lab['septic_patients']}", lab["label1_at_first_row"],
                     f"{lab['label1_rows_per_septic_min']} / {lab['label1_rows_per_septic_median']:.0f} / "
                     f"{lab['label1_rows_per_septic_max']}",
                     task["rows"], task["positive_rows"], f"{100 * task['row_prevalence']:.2f}%"])
    t1 = table(["Set", "Patients", "Hourly rows", "Septic patients", "Label monotone (0 then 1)",
                "Label 1 from first row", "Label 1 rows per septic patient (min / median / max)",
                "Task rows", "Positive task rows", "Row prevalence"], rows)
    sp = c["split_A_patients"]
    rows2 = []
    for k, label in (("train", "Training pool (set A, 60%)"), ("val", "Validation subset (set A, 20% split)"),
                     ("test_A", "Internal test subset (set A, 20% split)"), ("test_B", "External test subset (set B)")):
        e = c[f"eval_{k}"]
        rows2.append([label, e["patients"], e["septic_patients"], e["rows"], e["positive_rows"],
                      f"{100 * e['row_prevalence']:.2f}%"])
    t2 = table(["Partition", "Patients", "Septic patients", "Rows", "Positive rows", "Row prevalence"], rows2)
    return (t1 + "\n\nSet A patient split (all patients): " + f"{sp['train']} training pool, {sp['val']} validation, "
            f"{sp['test']} internal test.\n\n" + t2)


def block_grids(_):
    rows = []
    for m, g in GRIDS.items():
        n = int(np.prod([len(v) for v in g.values()]))
        rows.append([NAMES[m], "; ".join(f"{k} in {v}" for k, v in g.items()), n])
    return table(["Model", "Grid", "Configurations"], rows)


def block_headline(d):
    b = d["boot"]
    m = d["metrics"]
    rows = []
    for s, label in SETS.items():
        prev = m[(m["set"] == s)]["prevalence"].iloc[0]
        for model in MATCHED + ["ref_logreg_full", "ref_hgb_full"]:
            g = b[(b["set"] == s) & (b["comparison"] == model)].set_index("metric")
            if g.empty:
                continue
            rows.append([label, NAMES[model], ci(g.loc["sens_at_5fpr"]), ci(g.loc["auroc"]), ci(g.loc["auprc"]),
                         f"{prev:.3f}"])
    return table(["Test set", "Model", "Sensitivity at 5% FPR", "AUROC", "AUPRC", "AUPRC baseline (prevalence)"],
                 rows)


def block_paired(d):
    b = d["boot"]
    rows = []
    for s, label in SETS.items():
        for comp in sorted(c for c in b["comparison"].unique() if "minus" in c):
            g = b[(b["set"] == s) & (b["comparison"] == comp)].set_index("metric")
            a, c = comp.split(" minus ")
            rows.append([label, f"{NAMES[a]} minus {NAMES[c]}", ci(g.loc["sens_at_5fpr"]), ci(g.loc["auroc"]),
                         ci(g.loc["auprc"])])
    return table(["Test set", "Difference", "Sensitivity at 5% FPR", "AUROC", "AUPRC"], rows)


def block_deployment(d):
    m, n = d["metrics"], d["headline"]["headline_N"]
    ref = d["ref"]
    rows = []
    frames = pd.concat([m[m["N"] == n], ref]) if ref is not None else m[m["N"] == n]
    for s, label in SETS.items():
        for model in MATCHED + ["ref_logreg_full", "ref_hgb_full"]:
            g = frames[(frames["set"] == s) & (frames["model"] == model)]
            if g.empty:
                continue
            for spec in (95, 90):
                rows.append([label, NAMES[model], f"{spec}%",
                             f3(g[f"spec{spec}_false_alarm_rate"].mean()), f3(g[f"spec{spec}_sensitivity"].mean()),
                             f3(g[f"spec{spec}_patients_detected"].mean()),
                             f"{g[f'spec{spec}_false_alarms_per_100_patient_days'].mean():.1f}"])
    return table(["Test set", "Model", "Validation specificity target", "Realised false alarm rate",
                  "Row sensitivity", "Septic patients detected", "False alarms per 100 non septic patient days"],
                 rows)


def block_learning(d):
    m = d["metrics"]
    rows = []
    for s, label in SETS.items():
        for model in MATCHED:
            g = m[(m["set"] == s) & (m["model"] == model)].groupby("N")["sens_at_5fpr"]
            cells = [f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(g.mean(), g.std(ddof=0))]
            rows.append([label, NAMES[model]] + cells)
    ns = sorted(m["N"].unique())
    return table(["Test set", "Model"] + [f"N = {n}" for n in ns], rows) + \
        "\n\nValues are mean ± standard deviation over the draw seeds."


def block_params(d):
    t = d["tuning"]
    n = d["headline"]["headline_N"]
    rows = []
    for model in MATCHED:
        g = t[(t["model"] == model) & (t["N"] == n)]
        rows.append([NAMES[model], "; ".join(f"seed {r.seed}: {r.params}" for r in g.itertuples()),
                     f"{g['cv_auroc'].mean():.3f}"])
    return table(["Model", "Selected configuration per draw", "Mean best CV AUROC"], rows)


def block_resources(d):
    r = d["res"]
    rows = [[x.map, x.reps, x.circuit, x.qubits, x.depth, x.two_qubit_gates, x.total_gates] for x in r.itertuples()]
    return table(["Map", "Repetitions", "Circuit", "Qubits", "Depth", "Two qubit gates (CX)", "Total gates"], rows)


def block_efficiency(d):
    e = d["eff"]
    rows = []
    for x in e.sort_values(["N", "model"]).itertuples():
        exact = "" if pd.isna(getattr(x, "circuits_exact_states", np.nan)) else f"{int(x.circuits_exact_states):,}"
        pair = "" if pd.isna(getattr(x, "circuits_sampled_pairwise", np.nan)) else \
            f"{int(x.circuits_sampled_pairwise):,}"
        rows.append([NAMES[x.model], x.N, f"{x.tune_seconds:.1f}", f"{x.train_seconds:.2f}",
                     f"{x.predict_ms_per_1000_rows:.1f}", f"{x.support_vectors:.0f}" if x.support_vectors >= 0 else "",
                     exact, pair])
    return table(["Model", "N", "Tuning time (s)", "Final fit time (s)", "Prediction time per 1000 rows (ms)",
                  "Support vectors", "Circuits, exact states (N plus evaluation rows)",
                  "Circuits, sampled pairwise estimator"], rows) + \
        f"\n\nEvaluation rows per fitted model (validation plus both test subsets): {int(e['eval_rows'].iloc[0]):,}."


def block_shots(d):
    s = d["shots"]
    if s is None:
        return "The shot noise study was not run."
    rows = [[x.shots, f3(x.sens_at_5fpr), f3(x.auroc), f3(x.auprc), f"{x.min_eigenvalue_before_projection:.4f}",
             x.negative_eigenvalues_clipped] for x in s.itertuples()]
    return table(["Shots per entry", "Sensitivity at 5% FPR (set A test)", "AUROC", "AUPRC",
                  "Smallest Gram eigenvalue before projection", "Negative eigenvalues clipped"], rows)


def block_runtime(d):
    i = d["info"]
    return (f"Full run wall clock time: {i['total_seconds'] / 60:.1f} minutes on the machine described above "
            f"(sizes {i['sizes']}, draw seeds {i['seeds']}, {i['bootstrap_resamples']} bootstrap resamples).")


BLOCKS = {"summary": block_summary, "data": block_data, "grids": block_grids, "headline": block_headline,
          "paired": block_paired, "deployment": block_deployment, "learning": block_learning,
          "params": block_params, "resources": block_resources, "efficiency": block_efficiency,
          "shots": block_shots, "runtime": block_runtime}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    args = ap.parse_args()
    res = Path(args.results)
    d = load(res)
    text = Path(args.readme).read_text(encoding="utf-8")
    for name, fn in BLOCKS.items():
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.S)
        if not pattern.search(text):
            continue
        try:
            body = fn(d)
        except (TypeError, KeyError, IndexError, AttributeError) as exc:
            body = f"Not available: the results files for this block are missing ({type(exc).__name__})."
        text = pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    Path(args.readme).write_text(text, encoding="utf-8")
    from sepsis6q import plots
    plots.make_all(res)
    print("README tables rendered from", res)


if __name__ == "__main__":
    main()
