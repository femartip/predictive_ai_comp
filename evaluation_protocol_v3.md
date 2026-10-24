# Evaluation protocol v3: simulated CodaBench on the AI Measurement Data Bank

Status: agreed with the team on 2026-10-21 (sections 3, 7, 8 and 10 revised after review). Replaces protocol v2 (`evaluation_protocol.md`) once the full bank is converted.

**Goal.** One offline procedure on which every approach (prior, test-time model, acquisition rule) is scored the way
CodaBench scores it, under the conditions we expect there:

- a few subject–benchmark pairs per run;
- benchmarks we have never trained on;
- most likely, subjects (models) we have never trained on.

It should be precise enough to give a stable ranking of approaches, which single CodaBench submissions cannot do:
the same predictions scored 0.232 and 0.304 at budget 0 on two submissions.

---

## 1. What CodaBench does (the target)

From the starting-kit README:

- **Eligible pairs** have at least 80 distinct items.
- **Draw.** The platform selects complete (benchmark, subject) groups under a limit of 1,000 distinct source items,
  "with preference for benchmarks and subjects represented less often in the draw and random tie-breaking". Groups
  that no longer fit are skipped.
- **Split.** Each pair has a persistent random 50/50 split into an acquisition pool and an evaluation pool. Repeated
  runs of an item stay in the same pool; an odd item goes to acquisition.
- **Acquisition.** Candidates arrive one at a time in a fixed random order. The hook sees the candidate, the current
  prediction, all labels bought so far in the draw (all pairs) and a context; it says buy or skip, and a skip is
  permanent. At most 31 labels per pair. Checkpoints at 0, 1, 3, 7, 15 and 31 labels; every pair finishes a
  checkpoint before any pair gets more labels.
- **Inputs.** `predict()` sees only the subject's 8 fields (`normalized_name`, `provider`, `release_date`,
  `access_date`, `harness`, `reasoning_effort`, `harness_version`, `subject_features_extra`), the item's 4 fields
  (`item_content`, `item_features`, `interactors`, `benchmark_id`: an anonymous alias), and `labeled`.
- **Score.**
  - Brier on every response run of the evaluation items, per pair and budget;
  - Brier ALC = the weighted mean over budgets (0.1, 0.2, 0.2, 0.2, 0.2, 0.1);
  - leaderboard = the unweighted mean over pairs;
  - ECE: 10 fixed bins, count-weighted, within each pair.

From our five formative feedbacks (41 pairs):

| | observed |
|---|---|
| benchmarks per draw | 7 in every draw (the formative set has 7, so every benchmark appears every time) |
| pairs per draw | 8 or 9 (29 benchmarks drawn with one subject, 6 with two) |
| source items per draw | 910–984 (the 1,000 cap is nearly filled) |
| items per pair | 88–226, **fixed per benchmark** (e.g. `benchmark_119137`: 44 evaluation items every time); every subject answers the whole benchmark |
| subjects | 40 distinct in 41 pairs: a subject almost never comes back |

So the hidden benchmarks are small and complete, and the subject pool is large.

---

## 2. Data

**Training data: the bank only** (team decision, 2026-10-21: the guidelines restrict training to the data bank; the
ADeLe battery is not used).

**Requirements on the converted bank** (the teammate's conversion script; one JSONL per benchmark, as in
`scripts/preprocess/*.py`):

- one row per (subject, item) with the mean score and the number of runs (or one row per run);
- the subject's 8 competition fields, filled where the bank has them;
- the item's text and features;
- a flag for responses that are not model outcomes: scores outside [0, 1], infrastructure failures, missing grades.
  These are dropped before anything else.

**Evaluable benchmarks.** A benchmark is evaluable if:

- its outcomes are binary (at least 90% of its responses are 0/1, as for MathArena);
- at least 3 subjects each answer at least 80 of its items.

Other benchmarks (partial credit, ratings, small ones) are **training only**.

**Fixed benchmark size (D1).** Each evaluable benchmark gets, once per repetition (section 3), a fixed random subset
of items whose size is drawn from the formative group sizes (88–226 items; the empirical distribution of the 41
observed groups). Only its subjects that answered all items of the subset can be drawn. A benchmark smaller than
its drawn size keeps all its items.

*Motivation:* hidden benchmarks are small and fixed. Today's simulator cuts large benchmarks into a different slice
in every draw, which makes groups that CodaBench never produces.

---

## 3. Partitions: what is "unseen"

Two partitions, crossed:

**Benchmark folds (D2): K_b = 10.**

- Benchmarks that share items, harness or model population are **siblings** and always fall in the same fold.
- Siblings are found two ways:
  1. name rules: `medhelm_*`; `helm_*` with `bbq`; `taubench` with `tau2_bench`;
  2. item overlap: two benchmarks sharing at least 1% of their items (exact match of normalised item text) are
     joined, transitively.
- Sibling groups are spread over 10 folds of similar size, balanced by the number of evaluable pairs.

**Model folds (D3): K_m = 5.**

- Subjects are grouped by `normalized_name`: every subject of a model, whatever its harness or reasoning effort,
  falls in the same fold.
- A subject without `normalized_name` is its own group.
- Groups are spread over 5 folds balanced by number of pairs.

**Cells.** Cell (k, j) = benchmark fold k × model fold j: 50 cells.

**Repetitions (D4): R = 5.** One repetition is one particular grouping of benchmarks and models into folds; a result
can hinge on which benchmarks share a fold. Each repetition redraws:

- both partitions;
- the fixed benchmark subsets (D1);
- the 50/50 splits;
- the acquisition orders.

Results average over repetitions, so no conclusion rests on one lucky partition. Quick exploration may run
repetition 1 alone; decisions use all five.

*Where CodaBench's variability comes from:* its split and each pair's acquisition order are fixed across
submissions; what changes is which subjects are drawn for the benchmarks. Redrawing splits and orders here only covers
the fact that we do not know the platform's own split.

---

## 4. Two regimes

**Primary: unseen benchmark, unseen model.** This is the regime closest to the competition; decisions follow it.

- For cell (k, j), the prior is trained on every pair whose benchmark is **not** in fold k **and** whose model is
  **not** in fold j. Training-only benchmarks follow the same rule.
- Draws for cell (k, j) use only benchmarks of fold k and subjects of fold j.
- Nothing in the draw, neither benchmark nor model, was in training.

**Secondary: unseen benchmark, known model.**

- For benchmark fold k, the prior is trained on everything outside fold k.
- Draws use benchmarks of fold k with any subjects.
- This covers the case where hidden subjects include models from the public bank.

Both are always reported.

*Motivation:* the team expects both benchmarks and subjects to be new. Holding out by `normalized_name` keeps
families and providers informative (a new GPT version shares its family with older ones), which is the realistic
case and what model-based routing needs.

---

## 5. Draws (D5)

Per repetition and regime:

- **1,000 evaluation draws** (seed 0), spread evenly over the cells: 20 per cell in the primary regime, 100 per
  benchmark fold in the secondary;
- **300 tuning draws** (seed 100), spread the same way.

One draw, in its cell:

1. **Candidate groups:** the (benchmark, subject) pairs of the cell whose subject answered the whole fixed subset of
   the benchmark.
2. **Fill greedily** up to 1,000 distinct source items, offering all the fold's benchmarks (about 10):
   - repeatedly add a group that still fits;
   - choose the one whose benchmark, then whose subject, is least represented in the draw so far;
   - break ties at random;
   - stop when no group fits.

   This is the platform's rule. With groups of 88–226 items, the cap stops it at about 6–10 benchmarks with mostly
   one subject each: more when the drawn benchmarks are small, fewer when they are large. The formative set always
   shows 7 only because it has 7. Every approach is scored on the same draws, so the variable draw size keeps
   comparisons paired, and results can be broken down by draw size.
3. **Split and order:** each pair uses its persistent 50/50 split and its fixed acquisition order (D4).

**Checks** (printed with every run): pairs per draw, benchmarks per draw, source items per draw and evaluation items
per pair, against the table of section 1.

---

## 6. Running an approach

**Fast path** (for exploring):

- prior logits are computed once per cell and cached;
- the test-time model and the acquisition rule run in numpy on those logits, with the platform's streaming
  semantics: candidates in order, buy/skip decisions, a refit after every purchase, checkpoints 0–31;
- this is `night_experiments/mixed.py`, adapted to cells.

**Faithful path** (before any submission):

- the real `model.py` (and `labeling.py`) runs inside the kit's `tools/streaming_ingestion.py` on 20 simulated draws
  per regime, from competition-format CSVs built from the bank (only the 8 + 4 input fields);
- its scores must match the fast path within 0.001;
- it also catches interface bugs.

**Input discipline.** In both paths an approach sees only what `predict()` sees:

- no benchmark names (an anonymous alias per draw);
- no subject IDs in features;
- no bank metadata beyond the competition fields.

Model attributes come from what the submission can obtain at runtime: Epoch AI or Hugging Face lookups, then the kNN
fallback.

---

## 7. Scores and outputs

Per (repetition, regime, approach), on the evaluation draws:

- **Brier and ECE per budget**: mean over pairs, with a ±1 s.e. band. These are the two plots the team uses, in the
  style of `submissions/feedback_*.png`.
- **Brier ALC and ECE ALC**: the leaderboard number, i.e. the mean over draws of each draw's mean over pairs.
- **Breakdowns:** by route (agentic, single-turn), by benchmark, by budget, and by draw size (benchmarks per draw).
- **Comparisons:** each approach against a reference on the **same draws**, as paired differences.

Brier with repeated runs counts each run as an outcome (n·p² − 2·p·s + q per pair, as now).

---

## 8. Statistics (D6)

- **Paired differences** between approaches, on identical draws, partitions and splits.
- **Cluster bootstrap over cells.** Draws within a cell share one trained prior and the same subjects, so they are not
  independent. Resample whole cells (50 per repetition, 250 over five), 2,000 times, and report the 95% interval.
- **A result counts** when its interval excludes zero.

---

## 9. Tuning (D7)

- Hyperparameters (α, temperatures, shrink factors, T3 scales…) are chosen on the **tuning draws** only, over small
  grids. The chosen values are reported with the result.
- The tuning draws use the same cells and priors as the evaluation draws. That is slightly optimistic, but the same
  for every approach, as in v2.
- Anything tuned on a breakdown (e.g. per route) uses the tuning draws of that breakdown.

---

## 10. Baselines

**Baselines in every run:**

| | what it is |
|---|---|
| B1 | 0.5 always |
| B2 | the running mean of the pair's revealed labels |
| B3 | R1 start (mean base rate of the training benchmarks of the route) + T1 |
| oracle level | the true pass rate of each pair at every budget: a floor on what any level-only method can reach |

**Checks of the draws** (section 5) are printed with every run. There is no calibration against CodaBench scores:
single submissions are too noisy for it (team decision).

---

## 11. Cost (cremaet, A5000)

Assumed: about 1.5M (subject, item) pairs, the current converted bank; more if the full bank is converted without
caps.

| step | count | time |
|---|---|---|
| booster fits per prior spec, primary regime | R × K_b × K_m × 2 routes = 500 | about 25–80 min on the GPU (3–10 s per fit; last night's 1.5M-pair fits took 2–3 s) |
| booster fits per prior spec, secondary regime | R × K_b × 2 = 100 | about 5–15 min |
| simulator, per approach | 5 repetitions × 2 regimes × 1,300 draws | about 25–40 min on 16 cores |
| faithful path | 2 × 20 draws | about 10 min |

A new prior costs about 1.5 hours end to end; a new test-time or acquisition rule on cached priors about 35 minutes.
Repetition 1 alone takes a fifth of that.

---

## 12. Open points

- **Implementation.** The script is written once the team's full conversion of the bank (JSONL) exists, and reads
  it directly.

- **Size of each benchmark fold.** About 10 evaluable benchmarks per fold with today's data. A fold with few small
  benchmarks gives smaller draws; the checks of section 5 will show it.
- **Sibling detection.** It needs item texts for every benchmark. Benchmarks that share no item text but share a
  harness and a model population (as HELM's do) are only caught by the name rules; worth reviewing by hand once.
- **Subjects with multiple settings.** A subject is model + settings, but the hold-out is by model, so all settings
  of a model leave together. That is stricter than CodaBench if the hidden subjects include known models with new
  settings; the secondary regime covers that side.
