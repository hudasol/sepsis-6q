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

from sepsis6q.classical import EXTENDED_ALL_GRIDS, EXTENDED_GRIDS, GRIDS  # noqa: E402

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

    out = {k: read(v) for k, v in {
        "counts": "data_counts.json", "metrics": "metrics_runs.csv", "ref": "reference_metrics.csv",
        "boot": "bootstrap.csv", "headline": "headline.json", "res": "resources.csv", "eff": "efficiency.csv",
        "shots": "shot_noise.csv", "tuning": "tuning.csv", "info": "run_info.json"}.items()}
    out["sens"] = load_sensitivity(res / "sensitivity")
    out["sens_all"] = load_sensitivity(res / "sensitivity_all")
    out["timing"] = read_file(res / "timings_clean.csv")
    out["ceiling"] = read_file(res / "ceiling_metrics.csv")
    out["ceiling_boot"] = read_file(res / "ceiling_bootstrap.csv")
    out["ceiling_info"] = read_file(res / "ceiling_info.json")
    return out


def read_file(path):
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


def load_sensitivity(folder):
    """Files of one post hoc sensitivity run, or None when that run is not present."""
    if not folder.is_dir():
        return None
    files = {"metrics": "metrics_runs.csv", "ref": "reference_metrics.csv", "boot": "bootstrap.csv",
             "headline": "headline.json", "tuning": "tuning.csv", "info": "run_info.json",
             "diag": "kernel_diagnostics.csv"}
    out = {k: read_file(folder / v) for k, v in files.items()}
    return out if out["boot"] is not None else None


def verdict(row):
    if row["ci_low"] > 0:
        return "higher, and the 95% interval excludes zero"
    if row["ci_high"] < 0:
        return "lower, and the 95% interval excludes zero"
    return "not distinguishable from it at this sample size (the 95% interval includes zero)"


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
    for s, label in SETS.items():
        ctrl = b[(b["set"] == s) & (b["comparison"] == "qsvm_zz minus qsvm_product")
                 & (b["metric"] == "sens_at_5fpr")].iloc[0]
        parts.append(f"Against its own product map control on the {label[0].lower() + label[1:]}, the entangling "
                     f"map was {verdict(ctrl)} (difference {ci(ctrl)}).")
    claim = any(b[(b["comparison"] == f"qsvm_zz minus {best}") & (b["metric"] == "sens_at_5fpr")]["ci_low"] > 0)
    parts.append("The prespecified question is answered " + ("yes on at least one test set." if claim else
                 "no: the quantum kernel did not detect more pre onset hours at the same false alarm rate."))
    draws = len(d["info"]["seeds"]) if d["info"] else d["tuning"]["seed"].nunique()
    extra = " " + posthoc_sentence(d) if d.get("sens") else ""
    return f"Headline result at N = {h['headline_N']} balanced training rows, mean over {draws} draws, " \
           f"patient level bootstrap with {h['resamples']} resamples. " + " ".join(parts) + extra


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
                     f"{x.predict_ms_per_1000_rows:.2f}", f"{x.support_vectors:.0f}" if x.support_vectors >= 0 else "",
                     exact, pair])
    return table(["Model", "N", "Tuning time (s)", "Final fit time (s)", "Prediction time per 1000 rows (ms)",
                  "Support vectors", "Circuits, exact states (N plus evaluation rows)",
                  "Circuits, sampled pairwise estimator"], rows) + \
        f"\n\nEvaluation rows per fitted model (validation plus both test subsets): {int(e['eval_rows'].iloc[0]):,}."


def block_shots(d):
    s = d["shots"]
    if s is None:
        return "The shot noise study was not run."
    rows = [[x.shots, f3(x.sens_at_5fpr), f3(x.auroc), f3(x.auprc), f"{x.min_eigenvalue_before_projection:.2e}",
             x.negative_eigenvalues_clipped] for x in s.itertuples()]
    return table(["Shots per entry", "Sensitivity at 5% FPR (set A test)", "AUROC", "AUPRC",
                  "Smallest Gram eigenvalue before projection", "Negative eigenvalues clipped"], rows)


def block_runtime(d):
    i = d["info"]
    return (f"Full run wall clock time: {i['total_seconds'] / 60:.1f} minutes on the machine described above "
            f"(sizes {i['sizes']}, draw seeds {i['seeds']}, {i['bootstrap_resamples']} bootstrap resamples).")


def block_runs(d):
    m = d["metrics"]
    top = m["N"].max()
    if d["ref"] is not None:
        m = pd.concat([m, d["ref"]])
    rows = [[NAMES[x.model], x.N, x.seed, x.set, f3(x.sens_at_5fpr), f3(x.auroc), f3(x.auprc), f3(x.prevalence)]
            for x in m.itertuples() if x.set != "val" and (x.N == top or x.model.startswith("ref"))]
    return table(["Model", "N", "Draw seed", "Set", "Sensitivity at 5% FPR", "AUROC", "AUPRC", "Prevalence"], rows)


def block_verdict(d):
    b, h, m = d["boot"], d["headline"], d["metrics"]
    best = h["best_matched_classical"]
    out = []
    for s, label in SETS.items():
        diff = b[(b["set"] == s) & (b["comparison"] == f"qsvm_zz minus {best}")].set_index("metric")
        primary = diff.loc["sens_at_5fpr"]
        claim = "an improvement can be claimed" if primary["ci_low"] > 0 else "no improvement can be claimed"
        out.append(f"* Primary question, {label[0].lower() + label[1:]}: the entangling QSVM minus "
                   f"{NAMES[best]} difference in sensitivity at 5% FPR was {ci(primary)}, so {claim}. "
                   f"AUROC difference {ci(diff.loc['auroc'])}, AUPRC difference {ci(diff.loc['auprc'])}.")
    for s, label in SETS.items():
        ctrl = b[(b["set"] == s) & (b["comparison"] == "qsvm_zz minus qsvm_product")
                 & (b["metric"] == "sens_at_5fpr")].iloc[0]
        tail = "" if ctrl["ci_low"] > 0 else " The data therefore do not support a claim that entanglement matters."
        out.append(f"* Entanglement control, {label[0].lower() + label[1:]}: entangling minus product map "
                   f"difference {ci(ctrl)}.{tail}")
    for s, label in SETS.items():
        g = m[m["set"] == s].groupby(["N", "model"])["sens_at_5fpr"].mean().unstack()
        parts = []
        for n in g.index:
            row = g.loc[n]
            top = row[["logreg", "rbf_svm", "hgb"]]
            parts.append(f"N = {n}: QSVM {row['qsvm_zz']:.3f} vs best classical {top.max():.3f} "
                         f"({NAMES[top.idxmax()]})")
        out.append(f"* Small data setting, {label[0].lower() + label[1:]} (mean over draws, descriptive only, "
                   f"no intervals): " + "; ".join(parts) + ".")
    return "\n".join(out)


def block_status(d):
    c, i = d["counts"], d["info"]
    files = f"{c['raw_A']['files']:,} set A files and {c['raw_B']['files']:,} set B files"
    if i["sizes"] == [250, 500, 1000, 2000] and len(i["seeds"]) == 3 and d["boot"] is not None:
        return f"Status: final results from the full run on {files}."
    return (f"Status: PRELIMINARY. The numbers below come from training sizes {i['sizes']} and draw seeds "
            f"{i['seeds']} on {files}, before the full run. They will be replaced by the full run.")


# Post hoc sensitivity analyses (README sections 7.8 and 7.8.1) and the at a glance block


def brow(b, set_name, comparison, metric="sens_at_5fpr"):
    return b[(b["set"] == set_name) & (b["comparison"] == comparison) & (b["metric"] == metric)].iloc[0]


def diff_rows(boot, headline):
    best = headline["best_matched_classical"]
    return best, {s: brow(boot, s, f"qsvm_zz minus {best}") for s in SETS}


def conclusion(prim, post):
    """What the paired differences under another grid say, relative to the primary analysis."""
    if any(r["ci_low"] > 0 for r in post.values()):
        return "an improvement can be claimed on at least one test set under these grids"
    if any(r["ci_high"] < 0 for r in post.values()):
        return "the QSVM remained lower than the best matched classical model on at least one test set"
    gone = [SETS[s][0].lower() + SETS[s][1:] for s, r in prim.items() if r["ci_high"] < 0]
    tail = f", and the deficit of the primary analysis on the {' and the '.join(gone)} did not persist" if gone else ""
    return "no improvement and no deficit can be claimed" + tail


def posthoc_runs(d):
    """(description, paired difference rows) for each post hoc run that is present."""
    runs = []
    for key, what, section in (("sens", "wider bandwidth and gamma grids", "7.8"),
                               ("sens_all", "every grid widened", "7.8.1")):
        if d.get(key):
            runs.append((what, section, diff_rows(d[key]["boot"], d[key]["headline"])[1]))
    return runs


def posthoc_sentence(d):
    """One sentence for the summary: the paired differences under the wider grids."""
    _, prim = diff_rows(d["boot"], d["headline"])
    runs = posthoc_runs(d)
    parts = [f"with {what} (section {sec}) the entangling QSVM minus best matched classical difference was "
             f"{ci(post['test_A'])} on the internal test and {ci(post['test_B'])} on the external test"
             for what, sec, post in runs]
    verdicts = {conclusion(prim, post) for _, _, post in runs}
    if len(verdicts) == 1:
        tail = f", so {verdicts.pop()}" if len(runs) == 1 else f"; in both cases {verdicts.pop()}"
    else:
        tail = "; " + "; ".join(f"under {what} {conclusion(prim, post)}" for what, _, post in runs)
    label = "Post hoc check" if len(parts) == 1 else "Post hoc checks"
    return f"{label}, added after the primary results were seen and leaving the prespecified answer " \
           "unchanged: " + "; ".join(parts) + tail + "."


def block_glance_answer(d):
    b, h = d["boot"], d["headline"]
    best, prim = diff_rows(b, h)
    q = {s: brow(b, s, "qsvm_zz")["estimate"] for s in SETS}
    c = {s: brow(b, s, best)["estimate"] for s in SETS}
    claim = any(r["ci_low"] > 0 for r in prim.values())
    text = (f"At {h['headline_N']} training rows the entangling QSVM reached a sensitivity of {q['test_A']:.3f} on the "
            f"internal test and {q['test_B']:.3f} on the external test at a 5% false positive rate, against "
            f"{c['test_A']:.3f} and {c['test_B']:.3f} for the best matched classical model ({NAMES[best]}), so the "
            f"prespecified question is answered {'yes' if claim else 'no'}")
    runs = posthoc_runs(d)
    if runs:
        diffs = ", and to ".join(f"{post['test_A']['estimate']:+.3f} and {post['test_B']['estimate']:+.3f} "
                                 f"with {what} (section {sec})" for what, sec, post in runs)
        verdicts = {conclusion(prim, post) for _, _, post in runs}
        tail = verdicts.pop() if len(verdicts) == 1 else "; ".join(conclusion(prim, post) for _, _, post in runs)
        text += (f"; post hoc reruns with wider tuning grids changed the paired differences to {diffs}, so "
                 f"{tail}")
    return text + "."


def block_glance_table(d):
    rows = []
    sources = [(NAMES[m], d["boot"], m) for m in MATCHED]
    if d.get("sens"):
        sources.append(("QSVM, ZZ entangling map, wider bandwidth grid (post hoc, section 7.8)", d["sens"]["boot"],
                        "qsvm_zz"))
    for label, b, model in sources:
        cells = [label]
        for s in SETS:
            cells += [ci(brow(b, s, model)), f3(brow(b, s, model, "auroc")["estimate"])]
        rows.append(cells)
    return table(["Model", "Internal test (set A): sensitivity at 5% FPR", "Internal test AUROC",
                  "External test (set B): sensitivity at 5% FPR", "External test AUROC"], rows)


def grid_text(g):
    return "; ".join(f"{k} in {v}" for k, v in g.items())


def block_sens_grids(d):
    rows = []
    for m in MATCHED:
        same = GRIDS[m] == EXTENDED_GRIDS[m]
        rows.append([NAMES[m], grid_text(GRIDS[m]), "unchanged" if same else grid_text(EXTENDED_GRIDS[m])])
    return table(["Model", "Primary grid", "Wider grid (this section)"], rows)


def block_sens_all_grids(d):
    rows = []
    for m in ("logreg", "hgb"):
        rows.append([NAMES[m], grid_text(GRIDS[m]), grid_text(EXTENDED_ALL_GRIDS[m])])
    return table(["Model", "Primary grid", "Widened grid (section 7.8.1)"], rows)


def block_sens_check(d):
    """Reproducibility of the new run: for runs whose selected configuration is unchanged, do the metrics agree?"""
    s, m0 = d["sens"], d["metrics"]
    key = ["model", "N", "seed"]
    cfg = d["tuning"][key + ["params"]].merge(s["tuning"][key + ["params"]], on=key, suffixes=("_p", "_s"))
    same_cfg = cfg[cfg["params_p"] == cfg["params_s"]]
    both = m0.merge(s["metrics"], on=key + ["set"], suffixes=("_p", "_s"))
    rows = []
    for model in MATCHED:
        runs = same_cfg[same_cfg["model"] == model]
        g = both.merge(runs[key], on=key)
        g = g[g["model"] == model]
        d_sens = (g["sens_at_5fpr_p"] - g["sens_at_5fpr_s"]).abs()
        d_auc = (g["auroc_p"] - g["auroc_s"]).abs()
        rows.append([NAMES[model], f"{len(runs)} of {len(cfg[cfg['model'] == model])}",
                     f"{int((d_sens.lt(1e-6) & d_auc.lt(1e-6)).sum())} of {len(g)}",
                     f"{max(d_sens.max(), d_auc.max()):.1e}" if len(g) else ""])
    ref = d["ref"].merge(s["ref"], on=["model", "set"], suffixes=("_p", "_s"))
    r_sens = (ref["sens_at_5fpr_p"] - ref["sens_at_5fpr_s"]).abs()
    r_auc = (ref["auroc_p"] - ref["auroc_s"]).abs()
    for model in ref["model"].unique():
        g = ref["model"] == model
        rows.append([NAMES[model], "not tuned", f"{int((r_sens[g].lt(1e-6) & r_auc[g].lt(1e-6)).sum())} of {int(g.sum())}",
                     f"{max(r_sens[g].max(), r_auc[g].max()):.1e}"])
    return table(["Model", "Runs (N and draw) with the same selected configuration as the primary run",
                  "Of those, rows (run and evaluation set) whose sensitivity at 5% FPR and AUROC agree to within 1e-6",
                  "Largest absolute difference among them"], rows) + (
        "\n\nThe two runs used different machines and operating systems. Logistic regression, boosted trees and "
        "the reference models reproduce exactly. Where the selected configuration is unchanged, the kernel models "
        "agree with the primary run to within the largest difference shown, which is consistent with rounding "
        "differences between processors in the single precision overlaps used for kernel prediction. All other "
        "differences come from new configurations chosen from the wider grid.")


def block_sens_params(d):
    s, n = d["sens"], d["sens"]["headline"]["headline_N"]
    rows = []
    for model in MATCHED:
        g = s["tuning"][(s["tuning"]["model"] == model) & (s["tuning"]["N"] == n)]
        rows.append([NAMES[model], "; ".join(f"seed {r.seed}: {r.params}" for r in g.itertuples()),
                     f"{g['cv_auroc'].mean():.3f}"])
    return table(["Model", "Selected configuration per draw (wider grids)", "Mean best CV AUROC"], rows)


def block_sens_headline(d):
    s = d["sens"]
    return block_headline({"boot": s["boot"], "metrics": s["metrics"]})


def block_sens_paired(d):
    s = d["sens"]
    best0, best1 = d["headline"]["best_matched_classical"], s["headline"]["best_matched_classical"]
    names = NAMES[best0] if best0 == best1 else f"primary {NAMES[best0]}, wider grids {NAMES[best1]}"
    rows = []
    for set_name, label in SETS.items():
        for what, c0, c1 in ((f"Entangling QSVM minus best matched classical model ({names})",
                              f"qsvm_zz minus {best0}", f"qsvm_zz minus {best1}"),
                             ("Entangling QSVM minus product map QSVM", "qsvm_zz minus qsvm_product",
                              "qsvm_zz minus qsvm_product")):
            rows.append([label, what, ci(brow(d["boot"], set_name, c0)), ci(brow(s["boot"], set_name, c1)),
                         ci(brow(s["boot"], set_name, c1, "auroc"))])
    return table(["Test set", "Difference at N = 2000", "Primary grids, sensitivity at 5% FPR",
                  "Wider grids, sensitivity at 5% FPR", "Wider grids, AUROC"], rows)


def block_sens_learning(d):
    return block_learning({"metrics": d["sens"]["metrics"]})


def block_diag(d):
    g = d["sens"]["diag"].sort_values(["model", "reps", "bandwidth"])
    rows = [[NAMES[x.model], x.reps, f"{x.bandwidth:g}", f"{x.mean_offdiag:.4f}", f"{x.var_offdiag:.2e}",
             f"{x.alignment:.4f}", f"{x.centered_alignment:.4f}"] for x in g.itertuples()]
    return table(["Map", "Repetitions", "Bandwidth", "Mean off diagonal entry", "Variance of off diagonal entries",
                  "Kernel target alignment", "Centered kernel target alignment"], rows)


def block_sens_all_headline(d):
    rows = []
    for set_name, label in SETS.items():
        for model in MATCHED:
            cells = [label, NAMES[model]]
            for src in (d["boot"], d["sens"]["boot"], d["sens_all"]["boot"]):
                cells.append(ci(brow(src, set_name, model)))
            rows.append(cells)
    return table(["Test set", "Model (N = 2000)", "Primary grids", "Wider kernel grids (section 7.8)",
                  "Every grid widened"], rows)


def block_sens_all_paired(d):
    s = d["sens_all"]
    rows = []
    for set_name, label in SETS.items():
        cells = [label]
        for src, hd in ((d["boot"], d["headline"]), (d["sens"]["boot"], d["sens"]["headline"]),
                        (s["boot"], s["headline"])):
            best = hd["best_matched_classical"]
            cells.append(f"{ci(brow(src, set_name, f'qsvm_zz minus {best}'))} ({NAMES[best]})")
        rows.append(cells)
    return table(["Test set", "Primary grids", "Wider kernel grids (section 7.8)", "Every grid widened"], rows)


def block_sens_all_params(d):
    s, n = d["sens_all"], d["sens_all"]["headline"]["headline_N"]
    rows = []
    for model in ("logreg", "hgb"):
        g = s["tuning"][(s["tuning"]["model"] == model) & (s["tuning"]["N"] == n)]
        rows.append([NAMES[model], "; ".join(f"seed {r.seed}: {r.params}" for r in g.itertuples())])
    return table(["Model", "Selected configuration per draw (every grid widened)"], rows)


# Clean timings and the classical ceiling reference (README sections 7.5 and 7.9)


def block_timing(d):
    t = d["timing"]
    g = t.groupby(["model", "N"]).agg(tune=("tune_seconds", "mean"), fit=("train_seconds", "mean"),
                                      pred=("predict_seconds", "mean"), rows=("predict_rows", "first"),
                                      sv=("n_support", "mean")).reset_index()
    order = {m: i for i, m in enumerate(MATCHED)}
    g = g.sort_values(["N", "model"], key=lambda col: col.map(order) if col.name == "model" else col)
    rows = [[NAMES[x.model], x.N, f"{x.tune:.1f}", f"{x.fit:.2f}", f"{x.pred / x.rows * 1e6:.2f}",
             f"{x.sv:.0f}" if x.sv >= 0 else ""] for x in g.itertuples()]
    h = t.iloc[0]
    return table(["Model", "N", "Tuning time (s)", "Final fit time (s)", "Prediction time per 1000 rows (ms)",
                  "Support vectors"], rows) + (
        f"\n\nMachine: {h['cpu_model']}, {h['logical_cores']} logical cores, {h['platform']}, Python {h['python']}, "
        f"numpy {h['numpy']}. Mean over {t['seed'].nunique()} draws, primary grids, one process with nothing else "
        f"running. Evaluation rows per fitted model: {int(t['predict_rows'].iloc[0]):,}.")


def block_ceiling(d):
    boot, c = d["ceiling_boot"], d["boot"]
    best = d["headline"]["best_matched_classical"]
    rows = []
    for s, label in SETS.items():
        g = boot[boot["set"] == s].set_index("metric")
        rows.append([label, "Boosted trees, all raw columns (ceiling reference, not matched)",
                     ci(g.loc["sens_at_5fpr"]), ci(g.loc["auroc"]), ci(g.loc["auprc"])])
        for name, comp, tag in ((NAMES["ref_hgb_full"], "ref_hgb_full", ""),
                                (f"{NAMES[best]}, N = {d['headline']['headline_N']} (matched)", best, "")):
            r = c[(c["set"] == s) & (c["comparison"] == comp)].set_index("metric")
            rows.append([label, name, ci(r.loc["sens_at_5fpr"]), ci(r.loc["auroc"]), ci(r.loc["auprc"])])
    return table(["Test set", "Model", "Sensitivity at 5% FPR", "AUROC", "AUPRC"], rows)


def block_ceiling_deployment(d):
    m = d["ceiling"]
    rows = []
    for s, label in SETS.items():
        g = m[m["set"] == s]
        for spec in (95, 90):
            rows.append([label, f"{spec}%", f3(g[f"spec{spec}_false_alarm_rate"].iloc[0]),
                         f3(g[f"spec{spec}_sensitivity"].iloc[0]), f3(g[f"spec{spec}_patients_detected"].iloc[0]),
                         f"{g[f'spec{spec}_false_alarms_per_100_patient_days'].iloc[0]:.1f}"])
    return table(["Test set", "Validation specificity target", "Realised false alarm rate", "Row sensitivity",
                  "Septic patients detected", "False alarms per 100 non septic patient days"], rows)


def block_ceiling_info(d):
    i = d["ceiling_info"]
    groups = ", ".join(f"{v} {k}" for k, v in i["feature_groups"].items())
    return (f"The ceiling model has {i['features']} features ({groups}) and was trained on all "
            f"{i['training_rows']:,} rows of the training pool.")


BLOCKS = {"status": block_status, "verdict": block_verdict, "runs": block_runs, "summary": block_summary, "data": block_data, "grids": block_grids, "headline": block_headline,
          "paired": block_paired, "deployment": block_deployment, "learning": block_learning,
          "params": block_params, "resources": block_resources, "efficiency": block_efficiency,
          "shots": block_shots, "runtime": block_runtime,
          "glance_answer": block_glance_answer, "glance_table": block_glance_table,
          "sens_grids": block_sens_grids, "sens_check": block_sens_check, "sens_params": block_sens_params,
          "sens_headline": block_sens_headline, "sens_paired": block_sens_paired,
          "sens_learning": block_sens_learning, "diag": block_diag, "sens_all_grids": block_sens_all_grids,
          "sens_all_headline": block_sens_all_headline, "sens_all_paired": block_sens_all_paired,
          "sens_all_params": block_sens_all_params, "timing": block_timing, "ceiling": block_ceiling,
          "ceiling_deployment": block_ceiling_deployment, "ceiling_info": block_ceiling_info}


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
