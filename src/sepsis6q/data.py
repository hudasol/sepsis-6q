"""Loading of the PhysioNet/CinC 2019 hourly records."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, RAW_DIR, VITALS

COLUMNS = VITALS + ["ICULOS", "SepsisLabel"]


def list_patient_files(set_name, limit=None):
    files = sorted((RAW_DIR / f"set{set_name}").glob("p*.psv"))
    return files[:limit] if limit else files


def load_set(set_name, limit=None, use_cache=True):
    """Return one DataFrame with a patient column and the needed raw columns.

    Rows keep their original hourly order within each patient.
    """
    files = list_patient_files(set_name, limit)
    cache = CACHE_DIR / f"set{set_name}_{len(files)}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    frames = []
    for path in files:
        df = pd.read_csv(path, sep="|", usecols=COLUMNS)
        df.insert(0, "patient", f"{set_name}_{Path(path).stem}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["SepsisLabel"] = out["SepsisLabel"].astype(np.int8)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(out, fh)
    return out


def label_behaviour(df):
    """Summarise how SepsisLabel behaves within each patient."""
    stats = {"patients": 0, "septic_patients": 0, "monotone_septic": 0,
             "label1_at_first_row": 0, "label1_rows_per_septic": []}
    for _, g in df.groupby("patient", sort=False):
        y = g["SepsisLabel"].to_numpy()
        stats["patients"] += 1
        if y.max() == 0:
            continue
        stats["septic_patients"] += 1
        first = int(np.argmax(y))
        if np.all(y[first:] == 1):
            stats["monotone_septic"] += 1
        if first == 0:
            stats["label1_at_first_row"] += 1
        stats["label1_rows_per_septic"].append(int(y.sum()))
    rows = np.array(stats.pop("label1_rows_per_septic") or [0])
    stats["label1_rows_per_septic_min"] = int(rows.min())
    stats["label1_rows_per_septic_median"] = float(np.median(rows))
    stats["label1_rows_per_septic_max"] = int(rows.max())
    return stats
