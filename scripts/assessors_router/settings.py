"""Constants and paths of the routed assessors (README.md). Everything else (data, protocol v3, the prior, the test-time
models, scoring) is imported from scripts/assessors_v3, so both pipelines share one methodology and the same draws.

Paths, all inside the repository:
    data/datasets/assessor_data/<build>/  the training data: one Parquet file per benchmark, built by
                                          scripts/preprocess/assessor_data.py (default: the newest build); loaded by
                                          bank.py into assessors_v3's Data object
    results/assessors_router/<run>/       records, routing tables, report (tables and figures); cache/ holds the
                                          fitted assessors (not versioned)
Import this module before any assessors_v3 module: it puts assessors_v3 on sys.path.
The module names of this folder differ from those of assessors_v3 on purpose: both folders are on sys.path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "scripts" / "assessors_v3"
sys.path.insert(0, str(V3))

DATASET_ROOT = ROOT / "data" / "datasets" / "assessor_data"
RESULTS = ROOT / "results" / "assessors_router"

# This server (vent: 48 cores shared, no LightGBM GPU build): threads for LightGBM and the simulator
N_JOBS = 16
DEVICE = "cpu"

# The v3 recommended prior: the v3 assessor, and the base of the tier assessors and of the general assessor
PRIOR = "R1+B0+G1+I2"

# --- model performance (README section 2) ----------------------------------------------------------------------------
K_TIERS = 3                # performance tiers (quantiles of the training models' strengths)
STRENGTH_SHRINK = 1.0      # strength = sum of rel / (benchmarks + STRENGTH_SHRINK): models seen on few benchmarks
                           # are pulled towards 0 (the average model)

# --- tier assessors (README section 3) -------------------------------------------------------------------------------
# A training benchmark enters a tier's assessor only if the tier has at least this many (subject, item) pairs on it:
# every benchmark gets the same total weight in the prior, so a handful of pairs would get a huge weight each
MIN_TIER_PAIRS = 100

# --- router (README section 4) ---------------------------------------------------------------------------------------
# Small data (one row per training subject): small, regularised trees
ROUTER_PARAMS = {"learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 5, "min_data_per_group": 5,
                 "cat_smooth": 10, "lambda_l2": 1.0, "feature_fraction": 0.8, "verbose": -1}
ROUTER_ROUNDS = 150
INNER_FOLDS = 5            # general assessor: cross-validation folds over the training benchmarks (train_values)
