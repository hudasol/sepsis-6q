"""Early warning task definition, past only features and the shared preprocessing."""

import numpy as np
import pandas as pd

from .config import CARRY_LIMIT_HOURS, FEATURES, LABEL_LEAD_HOURS, TREND_HOURS, TREND_SOURCES, VITALS


def add_features(df):
    """Forward fill vitals within each patient (past values only) and add trends.

    The trend feature is the change over the previous TREND_HOURS hours of the
    forward filled series. Nothing here looks at future rows.
    """
    out = df.copy()
    grouped = out.groupby("patient", sort=False)
    out[VITALS] = grouped[VITALS].ffill(limit=CARRY_LIMIT_HOURS)
    grouped = out.groupby("patient", sort=False)
    for col in TREND_SOURCES:
        out[f"{col}_d{TREND_HOURS}"] = out[col] - grouped[col].shift(TREND_HOURS)
    out["hour"] = grouped.cumcount()
    return out


def early_warning_rows(df):
    """Keep the rows that define the early warning task.

    For a septic patient, onset is the first row with SepsisLabel 1 plus
    LABEL_LEAD_HOURS hours. Rows at or after onset are dropped, because an alarm
    there is no longer an early warning. Remaining label 1 rows are positives.
    Rows of non septic patients are all kept as negatives.
    """
    hour = df.groupby("patient", sort=False).cumcount()
    first_pos = df["hour"].where(df["SepsisLabel"] == 1).groupby(df["patient"]).transform("min")
    keep = first_pos.isna() | (hour < first_pos + LABEL_LEAD_HOURS)
    out = df.loc[keep].copy()
    out["septic"] = out.groupby("patient", sort=False)["SepsisLabel"].transform("max").astype(np.int8)
    out["y"] = out["SepsisLabel"].astype(np.int8)
    return out.reset_index(drop=True)


class Preprocessor:
    """Median imputation and min max scaling to [0, 1], fitted on training rows only."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.median_ = np.nanmedian(X, axis=0)
        filled = np.where(np.isnan(X), self.median_, X)
        self.min_ = filled.min(axis=0)
        self.max_ = filled.max(axis=0)
        self.range_ = np.where(self.max_ > self.min_, self.max_ - self.min_, 1.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        filled = np.where(np.isnan(X), self.median_, X)
        return np.clip((filled - self.min_) / self.range_, 0.0, 1.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def feature_matrix(df):
    return df[FEATURES].to_numpy(dtype=float)
