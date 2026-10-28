import numpy as np
import pandas as pd

from sepsis6q.config import FEATURES, VITALS
from sepsis6q.features import Preprocessor, add_features, early_warning_rows, feature_matrix
from sepsis6q.splits import balanced_draw, split_patients, stratified_subset


def synthetic_raw(n_patients=200, seed=0):
    """Hourly records with the SepsisLabel convention of the challenge: 1 from six hours before onset."""
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_patients):
        hours = int(rng.integers(10, 40))
        df = pd.DataFrame(rng.normal(80, 10, size=(hours, len(VITALS))), columns=VITALS)
        df[df > 95] = np.nan
        df["ICULOS"] = np.arange(1, hours + 1)
        y = np.zeros(hours, dtype=np.int8)
        if p % 4 == 0:
            first = int(rng.integers(0, hours - 3))
            y[first:] = 1
        df["SepsisLabel"] = y
        df.insert(0, "patient", f"A_p{p:06d}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_early_warning_rows_stop_at_onset():
    raw = synthetic_raw()
    rows = early_warning_rows(add_features(raw))
    for pid, g in rows.groupby("patient"):
        r = raw[raw["patient"] == pid]
        if r["SepsisLabel"].max() == 1:
            first = int(np.argmax(r["SepsisLabel"].to_numpy()))
            assert g["hour"].max() < first + 6
            assert g["y"].sum() == min(6, len(r) - first)
        else:
            assert len(g) == len(r) and g["y"].sum() == 0


def test_features_use_past_values_only():
    raw = synthetic_raw()
    full = add_features(raw)
    pid = raw["patient"].iloc[0]
    cut = raw[raw["patient"] == pid].iloc[:8]
    truncated = add_features(pd.concat([cut, raw[raw["patient"] != pid]], ignore_index=True))
    a = full[full["patient"] == pid][FEATURES].iloc[:8].to_numpy()
    b = truncated[truncated["patient"] == pid][FEATURES].to_numpy()
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b))


def test_patient_ids_disjoint_across_splits():
    rows = early_warning_rows(add_features(synthetic_raw()))
    split = split_patients(rows)
    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == rows["patient"].nunique()
    sub = stratified_subset(rows, split["test"], 5, seed=1)
    assert set(sub) <= te


def test_splits_are_seeded():
    rows = early_warning_rows(add_features(synthetic_raw()))
    a, b = split_patients(rows, seed=3), split_patients(rows, seed=3)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_balanced_draw_comes_from_training_pool_only():
    rows = early_warning_rows(add_features(synthetic_raw()))
    split = split_patients(rows)
    train = rows[rows["patient"].isin(split["train"])].reset_index(drop=True)
    idx = balanced_draw(train, 40, seed=0)
    assert train["y"].to_numpy()[idx].sum() == 20
    assert set(train["patient"].to_numpy()[idx]) <= set(split["train"])


def test_transforms_fitted_on_training_rows_ignore_test_rows():
    rows = early_warning_rows(add_features(synthetic_raw()))
    split = split_patients(rows)
    train = feature_matrix(rows[rows["patient"].isin(split["train"])])
    test = feature_matrix(rows[rows["patient"].isin(split["test"])])
    pre = Preprocessor().fit(train)
    before = pre.transform(train)
    changed = test * 3.0 + 100.0
    pre2 = Preprocessor().fit(train)
    pre2.transform(changed)
    np.testing.assert_array_equal(pre2.transform(train), before)
    np.testing.assert_array_equal(pre.median_, pre2.median_)
    out = pre.transform(changed)
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_array_equal(pre.transform(test)[:5], pre.transform(test[:5]))
