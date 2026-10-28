"""Fixed settings of the study. Everything here was decided before any result existed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"

# Raw columns used by the matched comparison.
VITALS = ["HR", "Resp", "Temp", "SBP", "MAP", "O2Sat"]
TREND_SOURCES = ["HR", "Resp"]
TREND_HOURS = 6
# The eight prespecified features, in qubit order.
FEATURES = VITALS + ["HR_d6", "Resp_d6"]

# Forward fill carries a past value for at most this many hours.
CARRY_LIMIT_HOURS = 12
# SepsisLabel turns 1 this many hours before clinical onset.
LABEL_LEAD_HOURS = 6

SPLIT_SEED = 2026
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
# Fixed evaluation subsets (patients), stratified so that prevalence is preserved.
TEST_SEPTIC_PATIENTS = 300
VAL_SEPTIC_PATIENTS = 200

TRAIN_SIZES = [250, 500, 1000, 2000]
HEADLINE_N = 2000
DRAW_SEEDS = [0, 1, 2]
CV_FOLDS = 3
BOOTSTRAP_RESAMPLES = 500
TARGET_FPR = 0.05
VAL_SPECIFICITIES = [0.95, 0.90]

SHOT_COUNTS = [128, 512, 2048, 8192]
SHOT_N = 500
