# Assessor components: what each one is and does

This file explains every component of the v3 assessors, one by one. For each component it says the same seven things, in this order:

- **Part:** where it plugs in, as the prior's start, a prior feature, or a test-time term.
- **What it is.**
- **How it is computed:** exactly what the code does, with the file that does it.
- **Availability:** whether it exists for a model seen in training, a new model, and a new benchmark.
- **Why we added it:** the problem it was meant to solve.
- **Why it can help, and its limits.**
- **Evidence so far:** protocol v3, 5 repetitions, main regime (unseen benchmark and unseen model), unless stated otherwise.

The code is in `scripts/assessors_v3/`. Results are in the report *Assessor approaches on the data bank* (2026-10-26).

---

## 1. Words used throughout

| Term | Meaning |
|---|---|
| **subject** | A model *with its settings* (harness, reasoning effort). Two harnesses of the same model are two subjects. |
| **model** | The `normalized_name` of a subject. All subjects of one model share it. |
| **item** | One question or task of a benchmark. Each subject's repeated runs on an item are averaged into one row (`build_data.py`). |
| **pair** | (a) one subject on one item, a row of the training table; or (b) one subject on one benchmark, the unit CodaBench scores. The text says which. |
| **base rate** | A benchmark's share of solved subject–item rows: its overall level. |
| **logit** | log(p / (1 − p)). Everything is added on this scale, so probabilities stay between 0 and 1. logit 0 = 50%, +1 ≈ 73%, −1 ≈ 27%. |
| **budget** | How many labels have been revealed for a subject–benchmark pair: 0, 1, 3, 7, 15 or 31. |
| **seen / new model** | Whether the model's name appears in the training data. In the main evaluation regime every test model is new. |
| **hidden benchmark** | The benchmark being predicted, never in training. Always the case on CodaBench. |

**How an approach is written:** `prior | test-time`, for example `R1+B0+G1+I2 | T2`. The prior lists its components joined by `+`, and the final prediction is

```
logit p = prior(subject, item)  +  test-time correction(labels revealed so far)
```

The prior is learned offline and never sees the hidden benchmark's labels. The test-time part is refitted after every label.

---

## 2. Quick reference

| Code | Part | One line | New model? | New benchmark? | Status |
|---|---|---|---|---|---|
| R1 | prior start | start from the training benchmarks' base rate | yes | yes | keep, in every prior |
| K1 | prior start | start from the base rates of the most similar training benchmarks | yes | yes | no gain over R1, drop |
| R2 | prior feature | the model's strength across benchmarks | **no** (missing) | yes | small gain at most |
| S1 | prior feature | 11 facts about the model from its name and fields | yes | yes | small, helps with others |
| G1 | prior feature | how far the model's group lands above or below benchmark averages | yes | yes | best model feature |
| I0 | prior feature | 12 statistics of the item text | yes | yes | small |
| I2 | prior feature | the item's 18 ADeLe demand levels | yes | yes, needs annotation | main source of gain |
| B0 | prior feature | the benchmark's average demand profile | yes | yes, needs annotation | small, significant gain with G1 |
| T0 | test-time | no correction: the prior alone | — | — | reference only |
| T1 | test-time | corrections for benchmark, subject and item | yes | yes | good |
| T2 | test-time | T1 plus how much to trust the prior on each benchmark | yes | yes | best, use it |
| B1, B2, B3, oracle | baselines | always 0.5; running mean; R1 \| T1; the true level | — | — | always reported |

**Recommended default:** `R1+B0+G1+I2 | T2` (Brier ALC 0.2062). It is within 0.0003 of the best, `R1+B0+G1+S1+I0+I2 | T2` (0.2059), and simpler.

---

## 3. The prior: how the parts fit together

The prior is one LightGBM model (gradient-boosted trees), retrained for every evaluation cell (`prior.py`):

1. **Training rows:** subject–item pairs of the training benchmarks, at most 30,000 per benchmark (random sample), so large benchmarks do not dominate.
2. **Target:** the row's mean score, between 0 and 1, so several runs give fractions. Loss: cross-entropy.
3. **Weights:** min(runs, 4) per row, rescaled so every training benchmark weighs the same in total, then to mean 1.
4. **Start (R1 or K1):** every training row starts from its own benchmark's base rate (LightGBM's *init score*), so the trees only learn deviations from each benchmark's level.
5. **Features:** the columns produced by the chosen feature components (R2, S1, G1, I0, I2, B0), side by side.
6. **Prediction on a hidden benchmark:** the start (R1's global start, or K1's routed one) plus the trees' output.

Settings: 300 trees, learning rate 0.05, 31 leaves, at least 50 rows per leaf, 80% row and column sampling, L2 penalty 1 (`config.py`, `LGBM_PARAMS`).

**Why one tree model and not a formula:** the useful signal is in interactions ("small models fail when quantitative demand is high"), which trees find on their own.

---

## 4. Prior components

### R1 — start from the training benchmarks' base rate

- **Part:** prior start; also the start of baseline B3.
- **What it is:** the level each prediction begins from before any feature is used.
- **How it is computed** (`prior.py`): each training benchmark's base rate = sum of scores / sum of runs over its rows in the cell's training data.
  - *Training rows* start from their own benchmark's base rate.
  - *A hidden benchmark* starts from the mean of the training benchmarks' base rates, each benchmark counting once, converted to logit.
- **Availability:** always. It needs nothing from the hidden benchmark or the model.
- **Why we added it:** benchmark levels range from nearly 0% to nearly 100%. Without a start, the trees spend themselves learning those levels, which do not carry over to a new benchmark.
- **Why it can help, and its limits:**
  - *Helps:* it separates "how hard is this benchmark" (unknowable before labels) from "which items and models are above or below average" (learnable).
  - *Limit:* every hidden benchmark starts at the same level. This is the main reason the error at budget 0 is high: the level part of the budget-0 error is about 0.063 of 0.242.
- **Evidence:** protocol v1 showed that removing R1 was one of the worst changes. It is in every prior we keep.

### K1 — start routed by task type

- **Part:** prior start, replacing R1's start on hidden benchmarks; training rows still start from their own base rate.
- **What it is:** a guess of a hidden benchmark's base rate from the benchmarks that look most like it.
- **How it is computed** (`components.py`, `routed_start`):
  1. Each benchmark's profile = B0, its average ADeLe demands.
  2. The 18 profile values are standardised over the cell's training benchmarks.
  3. Take the 10 training benchmarks closest to the hidden one (Euclidean distance).
  4. The start = their logit base rates averaged with weights 1 / distance.
- **Availability:** any model; the hidden benchmark needs annotated items.
- **Why we added it:** budget 0 is a level problem (see R1). If math-heavy benchmarks tended to have low base rates, a new math-heavy benchmark should start low.
- **Why it can help, and its limits:** it only helps if a benchmark's demand profile predicts its base rate. It does not: two benchmarks with similar demands can have very different levels (different answer formats, graders, model populations).
- **Evidence:** alone (no trees, `start | T1`) it is no better than R1. Replacing R1 with K1 in the best prior changes nothing (+0.0002, 95% interval −0.0004 to +0.0008). **Drop it.**

### R2 — model strength

- **Part:** prior feature (one column, `strength`).
- **What it is:** one number per model: how good it is across benchmarks, after accounting for how hard each item is.
- **How it is computed** (`components.py`, `fit_rasch`, `R2`):
  1. On a sample of the training rows (at most 5,000 per benchmark), fit `logit P(success) = c + difficulty(item) + strength(model)` by penalised logistic regression. Items absorb benchmark difficulty, so strengths are comparable across benchmarks wherever models overlap.
  2. *Training rows* get a strength fitted **without their own benchmark group** (5 inner folds over sibling groups), so the feature looks as it would on a hidden benchmark.
  3. *Test rows* get the strength fitted on all training rows.
  4. 30% of the training models have their strength hidden (set to missing), so the trees learn what to do when it is missing.
- **Availability:** only for **models seen in training**. A new model has no strength: missing for every test row in the main regime, and likely for most CodaBench subjects.
- **Why we added it:** in protocol v1 the model's strength was the most useful thing the prior knew.
- **Why it can help, and its limits:** a strong model is likely to solve more items anywhere. But it is unavailable exactly where we need it (new models), and the hidden-strength trick only teaches the trees a fallback.
- **Evidence:** the previous default `R1+R2+S1+I0+I2` scores 0.2069, worse than the best prior without R2, `R1+B0+G1+S1+I0+I2` (0.2059; difference 0.0010, interval 0.0006 to 0.0015). In the known-model regime it helps a little more.

### S1 — system metadata

- **Part:** prior feature (11 columns).
- **What it is:** what the subject's own fields say about it, the information CodaBench gives with every subject.
- **How it is computed** (`components.py`, `system_table`), one row per subject:

  | Column | How |
  |---|---|
  | `provider` | first word of `normalized_name`, lower case (e.g. `openai`); a model table's `provider` if one is given |
  | `release_years` | release date in years since 2023, only with a model table (`ASSESSOR_MODELS_CSV`); missing otherwise, as now |
  | `access_years` | access date in years since 2023 |
  | `effort` | reasoning effort: none 0, minimal 0.5, low 1, medium 2, high 3, xhigh 4, max 5; 0 if not given |
  | `reasoning` | 1 if the name says *think*, *reason*, *R1*, *o1*, *o3*, *QwQ*…, or an effort is given, or the run thought |
  | `has_harness` | 1 if a harness is given |
  | `scaffold` | 1 if the harness looks like an agent scaffold (*OpenHands*, *Aider*, *Codex*, *SWE-agent*…) |
  | `log_params` | log10 of the parameter count when the name says it (e.g. *70B*), or from a model table |
  | `small_tier` | 1 if the name says *mini*, *flash*, *haiku*, *nano*, *lite*, *tiny*… |
  | `large_tier` | 1 if the name says *pro*, *opus*, *ultra*, *large*, *max*, *plus*, *turbo* |
  | `name_missing` | 1 if there is no `normalized_name` |

- **Availability:** **any model**, seen or new; it only reads the subject's fields.
- **Why we added it:** a new model still has a name, a provider and settings. Some of that should say how strong it is (a *mini* model is weaker than a *pro*; reasoning models do better on hard maths).
- **Why it can help, and its limits:** cheap and always available. But names are a noisy proxy for ability, and a provider's models differ hugely between generations.
- **Evidence:** provider and size are the model columns the trees use most. On their own they predict little on new benchmarks (AUC 0.60 in a quick test). Dropping S1 and I0 from the best prior costs a non-significant 0.0003.

### G1 — model group strength

- **Part:** prior feature (one column, `group_rel`).
- **What it is:** how far a model's *group* typically lands above or below a benchmark's average.
- **How it is computed** (`components.py`, `model_groups`, `G1`):
  1. Group = **provider × size tier × reasoning**, from S1: for example `openai|large|1`. Size tier = small if `small_tier`, large if `large_tier`, otherwise mid.
  2. On the training rows, for every (subject, benchmark): its success rate (smoothed: (successes + 1) / (items + 2)), as a logit, minus the benchmark's logit base rate. This is *rel*, how far above or below the benchmark's average the subject landed.
  3. Average rel per (group, benchmark), then per group over benchmarks, each benchmark counting once.
  4. *A training row* gets its group's average over the **other** training benchmarks (its own left out, as for a hidden benchmark).
  5. *A test row* gets the average over all training benchmarks.
- **Availability:** **any model**, including new ones: a new model needs only a group, which its name gives. Missing only for a group never seen in training.
- **Why we added it:** R2 is the right idea (model strength) but is missing for new models. G1 transfers it to groups, which new models belong to.
- **Why it can help, and its limits:** it gives new models a sensible level adjustment. But a group mixes models of different generations (an old and a new *pro* model share it), so it is a coarse estimate.
- **Evidence:** the single most useful model feature (6% of the trees' gain). With B0 it gains 0.0013 over `R1+I2` (interval −0.0018 to −0.0009).

### I0 — text statistics

- **Part:** prior feature (12 columns).
- **What it is:** simple, domain-free measurements of the item's text (`build_data.py`, `item_stats`).
- **How it is computed:** length in characters, words and lines (log scale); shares of digits, maths symbols (`\ $ ^ _ { } =`), code symbols and capitals; number of links and of question marks (log scale); whether it shows answer choices (`(A)`, `A)`); whether it is a placeholder (*see image*…); whether it has an image tag.
- **Availability:** **any model**, any benchmark; it needs only the text.
- **Why we added it:** before ADeLe, it was the only item information we had, and length or maths content plausibly relate to difficulty.
- **Why it can help, and its limits:** free and fast. But these are crude stand-ins for what ADeLe measures directly.
- **Evidence:** small; see S1.

### I2 — ADeLe demand levels

- **Part:** prior feature (18 columns, `<code>_exp`).
- **What it is:** how much an item demands of 18 abilities, each from 0 (none) to 5 (very high).
- **How it is computed:**
  1. A distilled annotator (Qwen2.5-7B with a LoRA adapter, `ceferra/qwen2.5-7b-adele-annotator`) reads the item once per rubric (`scripts/preprocess/annotations/`).
  2. From its probabilities for levels 0–5, the feature is the *expected* level: the sum of level × probability.
  3. At most 600 items per benchmark are annotated, chosen deterministically. Training and evaluation use annotated items only. Image items are read through their written image descriptions.

  | Code | Ability | Code | Ability |
  |---|---|---|---|
  | AS | attention and scan | KNs | knowledge of social sciences |
  | AT | atypicality | MCr | identifying relevant information |
  | CEc | verbal comprehension | MCt | critical thinking |
  | CEe | verbal expression | MCu | calibrating knowns and unknowns |
  | CL | conceptualisation, learning, abstraction | MS | mind modelling, social cognition |
  | KNa | applied sciences knowledge | QLl | logical reasoning |
  | KNc | everyday knowledge | QLq | quantitative reasoning |
  | KNf | formal sciences knowledge | SNs | spatial and physical reasoning |
  | KNn | natural sciences knowledge | VO | volume (how much to process) |

- **Availability:** **any model**. For a hidden benchmark, its items must be annotated at prediction time: **the annotator must run inside the submission** (still to confirm on CodaBench).
- **Why we added it:** the item side of the prior was the weak link. ADeLe describes *what* an item demands in the same terms for every benchmark, so the trees can carry "high quantitative demand is hard for this kind of model" from one benchmark to another.
- **Why it can help, and its limits:** it is the only item description that transfers across benchmarks. But demand levels describe items, not a benchmark's level, so they cannot fix the budget-0 level error. Above 4,096 tokens the annotator is less validated.
- **Evidence:** the main source of gain. ADeLe alone (`R1+I2`, 0.2075) gives two-thirds of the best prior's gain over B3 (0.2108). Item demands make up 38% of the trees' gain in the best prior.

### B0 — benchmark demand profile

- **Part:** prior feature (18 columns, `bench_<code>`).
- **What it is:** the average of the 18 demand levels over a benchmark's annotated items: one profile per benchmark, the same for all its rows.
- **How it is computed** (`components.py`, `bench_profiles`, `B0`): the mean of each `<code>_exp` over the benchmark's annotated items (at most 600). For a hidden benchmark: its own items' demands only, never its labels.
- **Availability:** **any model**; the hidden benchmark needs annotated items, like I2.
- **Why we added it:** to give the trees the *context* of an item. The same item demands mean something different in a maths benchmark than in a reading benchmark, and models of different strength react differently to whole task types.
- **Why it can help, and its limits:** it enables "this kind of model on this kind of benchmark" interactions. But it does not predict the benchmark's level (see K1); it works through interactions.
- **Evidence:** with G1, a gain of 0.0013 over `R1+I2`; 24% of the trees' gain in the best prior.

---

## 5. Test-time components

All test-time models are fitted on **every label revealed so far in the draw**, across all of its subject–benchmark pairs, and refitted after each new label (`testtime.py`). Labels are bought in the platform's fixed order, with no acquisition rule. Each pair keeps a persistent 50/50 split: labels come from one half, scores from the other.

### T0 — the prior alone

- **Part:** test-time (none). **What it is:** no correction; the prior's probability at every budget.
- **Why it is there:** to see how good the prior is on its own, and how much the test-time part adds.
- **Evidence:** about 0.242 at every budget for all priors. That is better than 0.5 everywhere (0.250) but far from the oracle (0.182).

### T1 — three corrections, shrunk towards zero

- **Part:** test-time. **What it is:**

  ```
  logit p = prior + b[benchmark] + u[subject, benchmark] + v[item]
  ```

- **How it is computed:** maximum a posteriori fit (L-BFGS) on the revealed labels. Each correction has a Gaussian prior centred at 0:
  - **b, benchmark:** is the hidden benchmark easier or harder than the prior thinks? Scale τ_bench = 1.0.
  - **u, subject on this benchmark:** does this subject do better or worse here than predicted? Scale τ_subject, tuned (0.6–2.4).
  - **v, item:** did other subjects of the draw find this very item easier or harder? Scale τ_item, tuned (1–4). It only moves when another pair of the same benchmark revealed that item.
- **Availability:** any model, any benchmark; it only needs labels.
- **Why we added it:** the prior cannot know a hidden benchmark's level or a new subject's level there; the first labels can. Shrinkage avoids overreacting to 1–3 labels.
- **Why it can help, and its limits:**
  - *Helps:* it fixes level errors quickly.
  - *Limit:* it can only move a whole benchmark, a whole pair or a seen item. It cannot change how the prior ranks the items it has not seen yet.
- **Evidence:** B3 (R1 | T1, no trained prior) already goes from 0.249 at budget 0 to 0.186 at budget 31.

### T2 — T1 plus how much to trust the prior

- **Part:** test-time. **What it is:**

  ```
  logit p = start + λ[benchmark] × (prior − start) + b + u + v
  ```

- **How it is computed:**
  1. The prior is split into its start (R1 or K1) and its deviation from the start.
  2. λ, one per benchmark, scales that deviation. λ has a Gaussian prior centred at α with scale 0.5, so before any label λ = α.
  3. b, u and v are as in T1.
  4. Everything is fitted together by MAP.
- **Availability:** any model, any benchmark.
- **Why we added it:** on some hidden benchmarks the prior ranks items well; on others its ranking does not transfer. T1 cannot tell the difference; T2 learns it from the labels.
- **Why it can help, and its limits:**
  - *Helps:* it protects against a prior that does not transfer (λ falls towards 0) and rewards one that does (λ ≥ 1).
  - *Limit:* with few labels, λ stays near α.
- **Evidence:** slightly better than T1 with every prior; the best is 0.2059. Tuning always chose α = 1: shrinking the prior before labels never helped.

### Test-time scales and their tuning

- **What:** τ_subject, τ_item and α are chosen per repetition and regime on 300 *tuning* draws, separate from the 1,000 evaluation draws (`evaluate.py --tune`).
  - Grid: τ_subject ∈ {0.6, 1.2, 2.4}, τ_item ∈ {1, 2, 4}, α ∈ {0.25, 0.5, 1}.
  - Fixed: τ_bench = 1.0, λ's scale 0.5.
- **Why:** these scales decide how fast the corrections follow the labels, and the best value depends on the data. Choosing them on separate draws keeps the evaluation honest.
- **What tuning chose:** τ_subject 0.6–1.2, τ_item 1–2, α = 1.

---

## 6. Baselines and references

| Name | What it is | Why it is there | Main regime |
|---|---|---|---|
| B1 | always 0.5 | the score of knowing nothing | 0.2500 |
| B2 | running mean of the pair's revealed labels (0.5 before any) | the official baseline; erratic after 1 label, when it predicts 0 or 1 | 0.2474 |
| B3 | R1 \| T1: global start + T1, no trees | what test-time alone achieves: **the bar a prior must clear** | 0.2108 |
| start \| T1 | the prior's own start + T1, no trees | what a routed start (K1) alone achieves; equal to B3 with R1 | — |
| oracle | each pair's true success rate on its evaluation items, at every budget | the best a level-only guess can do; its Brier is the average of rate × (1 − rate), so it depends on *which* pairs are evaluated, and runs on different data have different oracles | 0.1816 |

---

## 7. What is common to every approach (not a component, but it changes results)

- **Data:** only ADeLe-annotated items; runs of a subject on an item averaged; test conditions pooled into the same item (`build_data.py`).
- **Evaluable benchmarks:** at least 90% binary outcomes and at least 3 subjects with 80 or more items. The others are training only.
- **Evaluation:** protocol v3 (`evaluation_protocol_v3.md`). Benchmark folds × model folds, a fixed item subset per benchmark, draws of complete groups up to 1,000 items, 5 repetitions. Paired comparisons with cluster-bootstrap intervals.
- **Comparing runs:** only on identical data and draws. Runs built from different annotation sets (e.g. 20 benchmarks against 182) are not comparable: their oracle, baselines and conclusions differ.
