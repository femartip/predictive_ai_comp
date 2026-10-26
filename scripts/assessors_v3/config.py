"""Paths and constants of the v3 assessor pipeline (data bank + ADeLe v1 annotations, evaluation_protocol_v3.md).

Every path can be overridden with an environment variable, so the same code runs on vent, cremaet or a laptop:

    ASSESSOR_JSONL_DIR    converter runs (scripts/preprocess/measurement_db.py), ':'-separated: <benchmark>.jsonl
    ASSESSOR_ADELE_DIR    ADeLe v1 tables, <run>/<benchmark>.parquet (run = no-assets, asset-image), one row per
                          annotated item (default: the repo's annotations/adele_v1 if it exists, else the shared
                          folder on vent)
    ASSESSOR_MODELS_CSV   optional model attributes (scripts/preprocess/epoch_models_csv.py), joined on normalized_name
    ASSESSOR_WORK         where build_data.py, evaluate.py and report.py write (default ~/assessors_v3_work)

Competition constants follow the starting kit's README (streaming evaluation and scoring) and our five formative
feedbacks (evaluation_protocol_v3.md, section 1).
"""

import os
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
            Path(__file__).resolve().parents[2])      # the repository (or two levels up outside a clone)

# Converter runs to read, ':'-separated: by default the no-assets and the asset-image runs on vent (a benchmark is
# taken from the first folder that has its JSONL)
JSONL_DIRS = [Path(p) for p in os.environ.get(
    "ASSESSOR_JSONL_DIR",
    "/data/datasets/measurement_db/no-assets/20261022T205359Z_b5a02e2a:"
    "/data/datasets/measurement_db/asset-image/20261023T110411Z_26fccf60").split(":") if p]
_REPO_ADELE = ROOT / "annotations" / "adele_v1"
ADELE_DIR = Path(os.environ.get("ASSESSOR_ADELE_DIR",
                                _REPO_ADELE if _REPO_ADELE.exists() else "/data/datasets/annotations/adele_v1"))
MODELS_CSV = Path(os.environ["ASSESSOR_MODELS_CSV"]) if os.environ.get("ASSESSOR_MODELS_CSV") else None
WORK = Path(os.environ.get("ASSESSOR_WORK", Path.home() / "assessors_v3_work"))
DATA = WORK / "data"          # build_data.py: pairs/, items.parquet, subjects.parquet, benchmarks.csv
RESULTS = WORK / "results"    # evaluate.py: one folder per run

# --- competition protocol -------------------------------------------------------------------------------------------
BUDGETS = (0, 1, 3, 7, 15, 31)                 # label budgets per subject-benchmark pair
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)   # Brier ALC = weighted mean of the six budgets' Brier scores
MIN_ITEMS = 80                                 # an eligible pair has at least 80 distinct items
DRAW_ITEM_CAP = 1000                           # a draw holds complete groups up to 1,000 distinct source items
ECE_BINS = 10                                  # [0, 0.1), ..., [0.9, 1]

# Evaluation items per pair seen in the formative feedback (value: how many of the 41 pairs). A hidden benchmark's
# group is twice that (the 50/50 split), +1 half of the time (an odd item goes to acquisition): 88-227 items (D1).
FORMATIVE_EVAL_ITEMS = {44: 9, 53: 1, 54: 3, 56: 5, 58: 6, 60: 6, 68: 2, 71: 2, 72: 1, 76: 3, 113: 3}

# --- protocol v3 design (sections 2-5) -------------------------------------------------------------------------------
EVAL_BINARY_SHARE = 0.9    # evaluable benchmark: >= 90% of its responses are binary (score_type) ...
EVAL_MIN_SUBJECTS = 3      # ... and >= 3 subjects answered >= MIN_ITEMS of its instances
K_BENCH = 10               # D2: benchmark folds (sibling groups kept together)
K_MODEL = 5                # D3: model folds (subjects grouped by normalized_name)
REPS = 5                   # D4: repetitions (folds, subsets, splits and orders redrawn)
EVAL_DRAWS, EVAL_SEED = 1000, 0       # D5: evaluation draws per repetition and regime
TUNE_DRAWS, TUNE_SEED = 300, 100      # D5: tuning draws (D7)
BOOTSTRAP = 2000           # D6: cluster-bootstrap resamples over cells

# Sibling benchmarks share items, harness or model population: always held out together (D2). These are the name
# rules; protocol.py also joins benchmarks sharing >= SIBLING_OVERLAP of their item texts (transitively).
SIBLING_OVERLAP = 0.01


def sibling_group(benchmark):
    if benchmark.startswith("medhelm_"):
        return "MedHELM"
    if benchmark.startswith("helm_") or benchmark == "bbq":
        return "HELM"
    if benchmark in ("taubench", "tau2_bench"):
        return "tau-bench"
    return benchmark


# --- data ------------------------------------------------------------------------------------------------------------
# ADeLe v1 rubric codes (columns <code>_exp = expected level in the annotation tables)
ADELE_CODES = ["AS", "AT", "CEc", "CEe", "CL", "KNa", "KNc", "KNf", "KNn", "KNs", "MCr", "MCt", "MCu", "MS", "QLl",
               "QLq", "SNs", "VO"]

# --- prior (prior.py) ------------------------------------------------------------------------------------------------
# Training rows: at most this many (subject, instance) pairs per benchmark, sampled once per fit. Benchmarks get equal
# total weight anyway, so more rows from a large benchmark mostly add cost (the bank has ~30M annotated pairs).
TRAIN_PAIRS_PER_BENCHMARK = int(os.environ.get("ASSESSOR_TRAIN_PAIRS", 30_000))
LGBM_PARAMS = {"objective": "cross_entropy", "learning_rate": 0.05, "n_estimators": 300, "num_leaves": 31,
               "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
               "reg_lambda": 1.0, "verbose": -1}
R2_PAIRS_PER_BENCHMARK = 5_000   # R2: the strength fits use a smaller sample (strength needs far fewer rows)
R2_INNER_FOLDS = 5         # R2: training rows get strengths fitted without their own inner benchmark fold
K1_NEIGHBOURS = 10         # K1: training benchmarks whose base rates give a hidden benchmark's routed start
R2_DROPOUT = 0.3           # R2: share of training models whose strength is hidden (see components.py)

# --- test-time (testtime.py) -----------------------------------------------------------------------------------------
# Prior scales (standard deviations, logit) of the MAP terms. Defaults = the values used in the night experiments;
# evaluate.py --tune picks them on the tuning draws instead (D7) from TAU_GRID.
TAU = {"bench": 1.0, "subject": 0.6, "item": 1.0, "alpha": 1.0, "slope": 0.5}
# alpha < 1 shrinks the prior toward its start before any label (T2 at budget 0 = start + alpha * (prior - start))
# (subject and item widened upwards on 2026-10-25: the first full run chose the largest values, 1.2 and 2.0)
TAU_GRID = {"subject": [0.6, 1.2, 2.4], "item": [1.0, 2.0, 4.0], "alpha": [0.25, 0.5, 1.0]}

# --- resources -------------------------------------------------------------------------------------------------------
# vent is shared: build_data.py and evaluate.py refuse to start below START_RAM_GB available and stop (state is kept,
# rerun resumes) when available memory falls below MIN_FREE_RAM_GB.
START_RAM_GB = float(os.environ.get("ASSESSOR_START_RAM_GB", 10))
MIN_FREE_RAM_GB = float(os.environ.get("ASSESSOR_MIN_FREE_RAM_GB", 5))
