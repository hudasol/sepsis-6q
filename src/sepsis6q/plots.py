"""Figures in a pink theme, with distinct line styles and markers so they stay readable in greyscale."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PEARL = "#FBF4F2"
INK = "#4A2F38"
ROSE = "#A3274F"
STYLE = {
    "logreg": {"color": "#C98DA0", "ls": ":", "marker": "s", "label": "Logistic regression"},
    "rbf_svm": {"color": "#7E5361", "ls": "-.", "marker": "^", "label": "RBF SVM"},
    "hgb": {"color": "#D86E92", "ls": (0, (5, 2)), "marker": "D", "label": "Gradient boosted trees"},
    "qsvm_zz": {"color": ROSE, "ls": "-", "marker": "o", "label": "QSVM, ZZ entangling map"},
    "qsvm_product": {"color": "#E9A6BC", "ls": (0, (1, 1)), "marker": "v", "label": "QSVM, product map control"},
}
REF_STYLE = {"ref_logreg_full": ("#B98A98", (0, (8, 3)), "Reference: logistic regression, full pool"),
             "ref_hgb_full": ("#6B3A4A", (0, (3, 3, 1, 3)), "Reference: boosted trees, full pool")}
SET_TITLE = {"test_A": "Internal test (set A)", "test_B": "External test (set B)"}


def _theme():
    plt.rcParams.update({
        "figure.facecolor": PEARL, "axes.facecolor": PEARL, "savefig.facecolor": PEARL,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.grid": True, "grid.color": "#F2D3DC", "grid.linewidth": 0.8,
        "font.size": 10, "legend.frameon": False,
    })


def learning_curves(out):
    m = pd.read_csv(out / "metrics_runs.csv")
    ref = pd.read_csv(out / "reference_metrics.csv") if (out / "reference_metrics.csv").exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for ax, set_name in zip(axes, ["test_A", "test_B"]):
        d = m[m["set"] == set_name]
        for model, st in STYLE.items():
            g = d[d["model"] == model].groupby("N")["sens_at_5fpr"]
            if g.ngroups == 0:
                continue
            mean, lo, hi = g.mean(), g.min(), g.max()
            ax.plot(mean.index, mean.values, color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"])
            ax.fill_between(mean.index, lo.values, hi.values, color=st["color"], alpha=0.07)
        if ref is not None:
            for name, (color, ls, label) in REF_STYLE.items():
                r = ref[(ref["model"] == name) & (ref["set"] == set_name)]
                if len(r):
                    ax.axhline(r["sens_at_5fpr"].iloc[0], color=color, ls=ls, lw=1.2, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d["N"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(SET_TITLE[set_name], color=ROSE)
        ax.set_xlabel("Training rows N (balanced)")
    axes[0].set_ylabel("Sensitivity at 5% false positive rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Learning curves: mean over draws, band shows min to max", color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)


def headline_intervals(out):
    b = pd.read_csv(out / "bootstrap.csv")
    b = b[b["metric"] == "sens_at_5fpr"]
    order = list(STYLE) + list(REF_STYLE)
    labels = {**{k: v["label"] for k, v in STYLE.items()}, **{k: v[2] for k, v in REF_STYLE.items()}}
    markers = {**{k: v["marker"] for k, v in STYLE.items()}, "ref_logreg_full": "P", "ref_hgb_full": "X"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, set_name in zip(axes, ["test_A", "test_B"]):
        d = b[b["set"] == set_name].set_index("comparison")
        for i, model in enumerate(order):
            if model not in d.index:
                continue
            r = d.loc[model]
            color = ROSE if model == "qsvm_zz" else "#B9788C"
            ax.errorbar(r["estimate"], i, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                        fmt=markers[model], color=color, ecolor=color, capsize=3, ms=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([labels[k] for k in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(SET_TITLE[set_name], color=ROSE)
        ax.set_xlabel("Sensitivity at 5% FPR, 95% bootstrap interval")
    fig.tight_layout()
    fig.savefig(out / "headline_intervals.png", dpi=150)
    plt.close(fig)


def shot_noise(out):
    path = out / "shot_noise.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    exact = s[s["shots"] == "exact"]
    finite = s[s["shots"] != "exact"].copy()
    finite["shots"] = finite["shots"].astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(finite["shots"], finite["sens_at_5fpr"], color=ROSE, ls="-", marker="o", label="Sampled kernel")
    ax.axhline(exact["sens_at_5fpr"].iloc[0], color="#7E5361", ls="--", label="Exact kernel")
    ax.set_xscale("log", base=2)
    ax.set_xticks(finite["shots"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Shots per kernel entry")
    ax.set_ylabel("Sensitivity at 5% FPR (set A test)")
    ax.set_title(f"Shot noise, entangling QSVM, N = {int(s['N'].iloc[0])}", color=ROSE)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "shot_noise.png", dpi=150)
    plt.close(fig)


def kernel_diagnostics(out):
    path = out / "kernel_diagnostics.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    panels = [("mean_offdiag", "Mean off diagonal kernel entry", "linear"),
              ("var_offdiag", "Variance of off diagonal entries", "log"),
              ("alignment", "Kernel target alignment", "linear")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (col, title, scale) in zip(axes, panels):
        for model, color, marker in (("qsvm_zz", ROSE, "o"), ("qsvm_product", "#E9A6BC", "v")):
            for reps, ls in ((1, "-"), (2, "--")):
                g = d[(d["model"] == model) & (d["reps"] == reps)].sort_values("bandwidth")
                label = f"{STYLE[model]['label']}, reps {reps}"
                ax.plot(g["bandwidth"], g[col], color=color, ls=ls, marker=marker, ms=5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale(scale)
        ax.set_xticks(sorted(d["bandwidth"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(title, color=ROSE, fontsize=10)
        ax.set_xlabel("Bandwidth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"Kernel diagnostics, training Gram matrix, N = {int(d['N'].iloc[0])}, draw seed "
                 f"{int(d['seed'].iloc[0])}", color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "kernel_diagnostics.png", dpi=150)
    plt.close(fig)


def make_all(out):
    _theme()
    learning_curves(out)
    if (out / "bootstrap.csv").exists():
        headline_intervals(out)
    shot_noise(out)
    kernel_diagnostics(out)
