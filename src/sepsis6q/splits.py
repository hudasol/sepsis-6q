"""Seeded patient level splits, evaluation subsets and balanced training draws."""

import numpy as np
import pandas as pd

from .config import SPLIT_SEED, TRAIN_FRAC, VAL_FRAC


def patient_table(rows):
    return rows.groupby("patient", sort=True)["septic"].max().reset_index()


def split_patients(rows, seed=SPLIT_SEED):
    """Split patients 60/20/20 into training pool, validation and internal test, stratified by sepsis."""
    pts = patient_table(rows)
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, grp in pts.groupby("septic"):
        ids = grp["patient"].to_numpy().copy()
        rng.shuffle(ids)
        n_tr = int(round(TRAIN_FRAC * len(ids)))
        n_va = int(round(VAL_FRAC * len(ids)))
        parts["train"].extend(ids[:n_tr])
        parts["val"].extend(ids[n_tr:n_tr + n_va])
        parts["test"].extend(ids[n_tr + n_va:])
    return {k: np.array(sorted(v)) for k, v in parts.items()}


def stratified_subset(rows, patients, n_septic, seed):
    """Pick a fixed random patient subset with n_septic septic patients and the natural ratio of the rest."""
    pts = patient_table(rows[rows["patient"].isin(patients)])
    sep = pts.loc[pts["septic"] == 1, "patient"].to_numpy()
    non = pts.loc[pts["septic"] == 0, "patient"].to_numpy()
    if len(sep) <= n_septic:
        return np.array(sorted(pts["patient"]))
    frac = n_septic / len(sep)
    rng = np.random.default_rng(seed)
    n_non = int(round(frac * len(non)))
    chosen = np.concatenate([rng.choice(sep, n_septic, replace=False), rng.choice(non, n_non, replace=False)])
    return np.array(sorted(chosen))


def balanced_draw(train_rows, n, seed):
    """Draw n training rows, half positive and half negative, from the training pool."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(train_rows["y"].to_numpy() == 1)
    neg = np.flatnonzero(train_rows["y"].to_numpy() == 0)
    half = n // 2
    if len(pos) < half:
        raise ValueError(f"only {len(pos)} positive rows available for N={n}")
    idx = np.concatenate([rng.choice(pos, half, replace=False), rng.choice(neg, half, replace=False)])
    rng.shuffle(idx)
    return idx
