# Assessors v3: data bank + ADeLe v1, evaluated with protocol v3

Trains a basic assessor on the AI Measurement Data Bank and scores it the way CodaBench does, following
[evaluation_protocol_v3.md](../../evaluation_protocol_v3.md). Approach notation (R1, R2, S1, I0, I2, T1, T2, B1–B3)
as in [approaches.md](../../approaches.md).

- **Training data:** the bank only (the ADeLe battery is not used), restricted to ADeLe-annotated items (at most 600
  per benchmark, [annotations/adele_v1](../../annotations/adele_v1)).
- **Basic assessor:** prior `R1+R2+S1+I0+I2` (LightGBM started from each benchmark's base rate) with test-time `T2`.
  Each component is a separate function in `components.py`, so any combination can be evaluated (`--prior R1+I2`, ...).
- **Evaluation:** fast path of protocol v3. That means:
  - benchmark folds × model folds (50 cells);
  - primary regime (unseen benchmark, unseen model) and secondary regime (unseen benchmark, known model);
  - fixed benchmark subsets, persistent splits, and draws filled greedily up to 1,000 items;
  - Brier and ECE ALC, with paired cluster-bootstrap comparisons.

## Files

| File | What it does |
|---|---|
| `config.py` | Paths (overridable by environment variables), competition and protocol constants, model settings |
| `build_data.py` | Converter JSONL + ADeLe tables → compact training tables (one row per subject × instance) |
| `data.py` | Loads those tables into integer-coded arrays (`Data`); marks evaluable benchmarks |
| `protocol.py` | Protocol v3: sibling groups, benchmark and model folds, fixed subsets, splits, draws, draw checks |
| `components.py` | The prior's input components R2, S1, I0, I2 (R1 is the start, in `prior.py`) |
| `prior.py` | LightGBM prior of one cell (CPU, OpenCL GPU or CUDA) |
| `testtime.py` | T1 and T2: MAP corrections refitted on the labels revealed so far |
| `simulate.py` | Streaming simulation of a draw; Brier, ECE and ALC |
| `evaluate.py` | Runs everything per repetition, regime and cell; resumable; caches the priors |
| `report.py` | Summary tables, per-budget plots, paired comparisons (also between two runs) |
| `export_adele.py` | Copies the published ADeLe tables into the repo, slimmed to the expected levels |
| `setup_env.sh`, `run.sh`, `requirements.txt` | Environment and per-server launcher (vent: CPU; cremaet: GPU) |

## Data decisions (build_data.py)

- **Runs:** repeated runs of a subject on an item under the same condition are averaged into one pair. The pair keeps
  its run count, score sum and sum of squares, so Brier still counts every run, as on the platform.
- **Test conditions:** pooled. All runs of a subject on an item count for the same item, whatever their test
  condition: in many benchmarks the condition only labels one subject's run.
- **Item filter:** only ADeLe-annotated items are kept. Ungraded responses and scores outside [0, 1] are dropped.
- **Evaluable benchmark:** at least 90% binary responses and at least 3 subjects with at least 80 instances each.
  Every other benchmark is training only.

## Running

Install once per server (creates `~/assessors_v3/.venv`):

```bash
bash scripts/assessors_v3/setup_env.sh
```

On **vent** (the bank is in `/data`). Use `tmux new -s assessors`, since the full protocol takes hours:

```bash
bash scripts/assessors_v3/run.sh vent build                       # ~/assessors_v3_work/data
bash scripts/assessors_v3/run.sh vent evaluate --quick            # smoke test: 1 repetition, 100 draws
bash scripts/assessors_v3/run.sh vent evaluate --tune             # protocol v3: 5 repetitions, both regimes
bash scripts/assessors_v3/run.sh vent report R1_R2_S1_I0_I2
```

On **cremaet**, LightGBM runs on the GPU if this build supports it there, otherwise on the CPU. cremaet has no copy of
the bank: run `build` on vent and copy `~/assessors_v3_work/data/` across. Then use the same commands with `cremaet`
instead of `vent`.

Other priors, compared on the same draws:

```bash
bash scripts/assessors_v3/run.sh vent evaluate --tune --prior R1+I2
bash scripts/assessors_v3/run.sh vent report R1_I2 --against R1_R2_S1_I0_I2 --predictor T2
```

Safety on shared servers: the scripts refuse to start below 10 GB of free memory (vent) and stop cleanly below 5 GB.
Rerun the same command to resume: finished benchmarks, cells and records are kept. Threads: 16 on vent at low
priority, 8 on cremaet.

## Outputs (`~/assessors_v3_work/results/<name>/`)

- `records_rep<r>_<regime>.parquet`: Brier and ECE per draw, pair, budget and predictor.
- Predictors:
  - **B1:** 0.5 always.
  - **B2:** running mean of the pair's revealed labels.
  - **B3:** R1 | T1.
  - **oracle:** each pair's true level.
  - **T0, T1, T2:** the prior alone, or with each test-time model.
- `report/summary.csv`, `per_budget.csv` and `paired.csv` (each predictor minus B3, with 95% cluster-bootstrap
  intervals). Also breakdowns by benchmark, draw size and route, and the per-budget Brier and ECE plots.
- `checks.json`: pairs, benchmarks and items per draw, to compare with the formative feedback (7 benchmarks, 8–9
  pairs, 910–984 items per draw).
- `tau.json`: the test-time scales chosen on the tuning draws (`--tune`). `importance.csv`: the prior's feature gains
  per cell.

## Adding a component

Write `f(ctx, train_rows, test_rows) -> (X_train, X_test)` in `components.py` and register it in `FEATURES`.
`ctx.d` is the `Data` object; the rows index `ctx.d.P`. Use only what the platform's `predict()` sees: the subject's
fields and the item's content, never subject ids or benchmark names.

## Cost (estimates for the full annotated bank, ~30M pairs)

- **Building the data:** reads every JSONL once (~290 GB) on vent; I/O-bound, about an hour or two.
- **One prior fit:** about 1–3 minutes (LightGBM on ~4M sampled rows, plus R2's six strength fits).
- **The full protocol:** 5 repetitions × (50 + 10) cells ≈ 300 fits, so about 10–15 hours on vent's CPUs. Use
  `--reps 1` for about 2–3 hours.
- **Test-time and draws:** a few minutes per repetition, more with `--tune` (18 settings × 2 models).
