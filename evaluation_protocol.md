# Evaluation protocol v2: measurement-db + ADeLe battery

Decided on 2026-10-15. This protocol replaces the measurement-db-only evaluation used for
every number in `results/assessor_report_v4.pdf` (protocol v1), which stays the reference for those
results. Code: `pipeline/battery.py` (battery loader) and `run_baselines_v2` in
`pipeline/experiments.py`.

## What the evaluation must estimate

The competition scores an assessor on **hidden benchmarks**: benchmarks not seen in training,
evaluated in draws of whole subject–benchmark groups (80–1,000 items per group, ≤ 1,000
distinct items per draw), with labels revealed at budgets 0, 1, 3, 7, 15 and 31 and scored by
Brier ALC. The subjects are only partly known: in measurement-db only 22 of 287 subject ids
recur across benchmarks.

Protocol v1 replayed this with leave-one-benchmark-out on measurement-db. It has one weakness
above all: only **4 benchmarks** can enter a draw (MMDocRAG and SWE-rebench groups exceed 1,000
items), so every conclusion about generalisation to new benchmarks rests on four cases.

## The new data and the problems it brings

The original ADeLe battery v1.0 (`adele-ai-evaluations/ADeLe_battery_v1dot0`, gated, CC BY-SA
4.0) adds 16,108 items in 20 sub-benchmarks, with binary results for 15 models
(`verification_final`, from the ADeLe-AIEvaluation repository). 13 sub-benchmarks satisfy the
80–1,000 rule and can be evaluated; with them the evaluation grows from 4 to 17 benchmarks.

But the battery differs from the competition setting in ways that would bias a naive evaluation:

1. **Sibling benchmarks.** The 13 drawable sub-benchmarks come from only 5 source collections.
   TimeBench contributes 6 and AGIEval 4. Siblings share format, construction and difficulty
   patterns, so a held-out sub-benchmark whose siblings are in training looks easier than a truly
   new benchmark.
2. **The same 15 models answer everything.** In every battery sub-benchmark the subjects are the
   same 15 models, so their strength is known from the training folds. The competition's test
   subjects are largely new.
3. **Less competition-like.** Short question-answer prompts, one run per item, older models
   (babbage-002, davinci-002), and no competition-format subject fields (only model names).
4. **Size imbalance.** 13 battery benchmarks against 4 measurement-db ones, and 20 battery
   sub-benchmarks against 6 measurement-db benchmarks in training.

## Decisions

### D1. Two evaluation sets, reported separately

- **Primary: measurement-db** — MathArena, Multi-SWE-bench, REAL, ResearchCodeBench.
- **Secondary: ADeLe battery** — the 13 drawable sub-benchmarks.
- Each set is summarised by its own macro mean (equal weight per benchmark). An overall score,
  when needed, is the **equal-weight mean of the two set means**, not the mean of 17 benchmarks.

*Motivation:* measurement-db is the most competition-like data and keeps every v2 number comparable
with v1. Pooling 17 benchmarks would let the battery decide the result 13 to 4 (problem 4) while
it is the less representative source (problem 3).

### D2. Battery: hold out a whole source collection

| fold (source collection) | evaluated sub-benchmarks (items) |
|---|---|
| AGIEval | Civil Service Examination (408), GRE & GMAT (203), LSAT (910), SAT (410) |
| TimeBench | Date Arithmetic (493), MCTACO (205), MenatQA (680), TempReason (657), TimeDial (340), TimeQA (727) |
| LiveBench | Math (173) |
| MedCalcBench | MedCalcBench (556) |
| SciBench | SciBench (355) |

When a fold is held out, all its sub-benchmarks leave training, including the non-drawable ones
of the same collection (LiveBench Data Analysis, Language and Reasoning leave with LiveBench Math).

*Motivation:* problem 1. A new competition benchmark has no sibling in our training data, so the
evaluated benchmark should not have one either.

### D3. measurement-db folds unchanged

Each measurement-db benchmark is still held out on its own, as in v1. Known imperfection:
Multi-SWE-bench and SWE-rebench are siblings (both GitHub issue resolution), and SWE-rebench stays
in training when Multi-SWE-bench is evaluated.

*Motivation:* comparability with all v1 results; changing the measurement-db folds would move
every primary number for reasons unrelated to the assessor.

### D4. Train on everything not held out, balanced per source collection

Training data = all of measurement-db and the battery minus the held-out fold. Each **source
collection** gets the same total weight: the six measurement-db benchmarks are six collections,
the battery's sub-benchmarks are grouped into its nine sources (AGIEval, ChemLLMBench, LiveBench,
MMLU-Pro, MedCalcBench, OmniMath, SciBench, TimeBench, TruthQuest).

The same rule sets B3's starting level: the mean over training collections of each collection's
base rate (mean outcome over its subject–item pairs).

*Motivation:* problem 4. Per-benchmark balance would give the battery 20 of 26 shares; per-item
weighting would let MMDocRAG (2,000 items × 2 conditions) and MMLU-Pro (5,431 items) dominate.
Collections are the natural unit: they are the unit held out (D2).

### D5. Battery: a cold-subject variant

Every battery result is reported twice:

- **standard** — training as in D4;
- **cold subjects** — additionally, every response of the evaluated subjects' base models is
  removed from training: the 15 battery models, and every measurement-db subject with the same
  base model (GPT-4o, o1-mini, o1, Llama 3.2 3B, DeepSeek R1 Distill Qwen 32B; 34 subject ids in
  total). For the baselines (computed before the battery's subject fields were filled in), the
  match was by name on the first four only; B3's cold starting level is unaffected in practice.

*Motivation:* problem 2. The standard variant tells us how the assessor does when the subjects are
well known; the cold variant approximates the competition's mostly new subjects. The truth for
a hidden benchmark lies between the two.

Consequence for the baselines: B1, B2 and R1 use no training data, so they are identical in both
variants. B3's starting level in the cold variant comes from measurement-db only (the battery has
no responses left once its 15 models are removed).

### D6. Tuning stays within the evaluation set

Adaptation strengths (τ, and α for v2) are chosen by grid search, per held-out benchmark:

- measurement-db benchmark: mean ALC on the first 8 draws of the **other measurement-db**
  evaluation benchmarks (as in v1);
- battery benchmark: mean ALC on the first 8 draws of the battery evaluation benchmarks **outside
  its fold**.

*Motivation:* otherwise the 13 battery benchmarks would dominate the tuning of measurement-db
benchmarks, and a benchmark's own siblings would select its hyperparameters (problems 1 and 4).

### D7. Uncertainty

- **Over draws** (as in v1): percentile bootstrap, 2,000 resamples of the 30 draws, paired across
  predictors. Reflects split and label randomness for the given benchmarks.
- **Over benchmarks** (new, battery only): percentile bootstrap over the 13 battery benchmarks.
  Reflects which benchmarks happen to be tested — the uncertainty that matters for hidden
  benchmarks, and that four measurement-db benchmarks are too few to estimate.

### D8. Unchanged from v1

- Competition rules: groups of 80–1,000 items; draws hold whole groups up to 1,000 distinct items;
  50/50 acquisition/evaluation split per group; budgets 0, 1, 3, 7, 15, 31; ALC weights 0.1, 0.2,
  0.2, 0.2, 0.2, 0.1. The 7 battery sub-benchmarks outside the 80–1,000 range (MMLU-Pro, OmniMath,
  ChemLLMBench, TruthQuest, Data Analysis, Language, Reasoning) are training data only.
- One benchmark per draw; 30 draws per benchmark; seed 0 (measurement-db draws are identical to
  v1).
- Baselines and ceiling as defined in v1: B1 constant 0.5, B2 empirical mean (official), B3
  label-only hierarchical, R1 oracle item difficulty.

### D9. Annotations (for the assessor runs, not the baselines)

- measurement-db: ADeLe v1 and v2 levels from our distilled annotator, clean run (32k window).
- Battery: v1 levels = the GPT-4o annotations shipped with the dataset (ADeLe's reference
  annotations); v2 levels = our annotator.
- As features, v1 levels are integers in both sources (our annotator's most likely level on
  measurement-db, matching GPT-4o's integer levels on the battery), so a tree cannot tell the
  sources apart by the levels' format; v2 levels are our annotator's expected level in both.
- The four flagged v2 rubrics (SNk, SNp, SPa, SPv) are excluded everywhere.

*Caveat:* the v1 levels come from two annotators across the sources, which can shift the meaning
of a level between them.

### D10. Battery subject information (for the assessor runs)

The battery gives model names only. The competition-format subject fields are filled in by hand
in `pipeline/battery.py` (`SUBJECTS`): normalized name (measurement-db's convention), provider,
release date (first public API release) and reasoning effort (low for o1). Five base models are
shared with measurement-db through identical names: GPT-4o, o1-mini, o1, Llama 3.2 3B and DeepSeek
R1 Distill Qwen 32B. Blank: access date (not reported by ADeLe; between January and March 2025),
harness and harness version (direct prompting). Uncertain: the GPT-3.5 Turbo snapshot ADeLe used,
and whether Llama 3.2 11B/90B are the Vision-Instruct releases (the only 11B/90B Llama 3.2 models).

## Open issues

- Draws still contain one benchmark at a time; the competition mixes hidden benchmarks.
- The 80–1,000 rule keeps four large, diverse battery sub-benchmarks (MMLU-Pro, OmniMath,
  ChemLLMBench, TruthQuest) out of evaluation. Subsampling them to 1,000 items per group would
  add them, at the cost of departing from the competition rule.
- The SWE sibling pair in measurement-db (D3).
- The dataset's terms ask not to reveal battery examples in plain text online: battery data stay
  in `data/adele_battery/` (gitignored) and on the cluster's `/data`.

## Baselines and ceiling under protocol v2

Brier ALC (lower is better), 30 draws per benchmark; computed by
`python -m pipeline.experiments --study baselines_v2` (results in `results/baselines_v2/`,
per benchmark in `summary_by_benchmark.csv`).

| | measurement-db (4) | ADeLe battery (13) | overall (mean of the two) |
|---|---|---|---|
| B1 constant 0.5 | 0.2500 | 0.2500 | 0.2500 |
| B2 empirical mean (official) | 0.2635 | 0.2139 | 0.2387 |
| B3 label-only hierarchical | 0.1979 | 0.1892 | 0.1936 |
| B3, cold subjects | — | 0.1893 | — |
| R1 oracle item difficulty (ceiling) | 0.1264 | 0.1534 | 0.1399 |

- measurement-db: B1, B2 and R1 are identical to protocol v1 (same draws, no training data). B3
  moves from 0.1984 to 0.1979, because its starting level now also uses the battery's collections.
- Battery: the official baseline is much stronger there (0.2139, better than a constant) because
  many battery sub-benchmarks are nearly saturated by the stronger models, so a subject's own mean
  is informative. The headroom between B3 and the ceiling is half that of measurement-db (0.036
  against 0.072).
- The cold-subject variant does not change B3 (its starting level barely moves); it will matter
  for trained assessors, whose subject features carry model strength.
- Benchmark-level uncertainty is large: B3's battery mean has a 95% interval of 0.169–0.209 over
  benchmarks, against 0.1885–0.1900 over draws. Comparisons between assessors on the battery should
  be paired by benchmark.
