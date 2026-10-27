# Routed assessors: one assessor per performance cluster

Status: proposal, for team review. Notation (R1, G1, I2, B0, T1, T2, B3) as in `approaches.md` and
`scripts/assessors_v3/COMPONENTS.md`; methodology as in `evaluation_protocol_v3.md`.

## 1. Idea

The v3 prior is a single assessor for every model, so it has to average how strong and weak models react to the same
item. Our results point the other way: what transfers best is the model's level (R1, R2, G1), and the same item demands
probably mean different things to a frontier model and to a small one.

The proposal:

- Group the models by how well they perform.
- Train one assessor per group (a *specialist*).
- At prediction time, send each subject to the specialist of its group:
  - a **known model** goes to its own group;
  - an **unknown model** is routed from its subject fields, using the fields that best tell the groups apart.

Expected gains:

- **Budget 0:** each specialist starts from its own group's level instead of one global start. This targets the level
  error, the largest part of the budget-0 error.
- **All budgets:** each specialist learns item effects (ADeLe demands, benchmark profile) for one kind of model, so
  items should be ranked better within a pair.

## 2. Performance clusters

- **Unit:** the model (`normalized_name`): every subject of a model falls in the model's cluster. Settings (harness,
  reasoning effort) are missing for most subjects, and anything missing in the bank will also be missing on CodaBench,
  so clustering by settings would rest on little data. Clustering by model also matches the model folds of protocol v3.
- **Performance measure:** a benchmark-adjusted strength, not a raw mean score. A raw mean depends on which benchmarks
  a model happened to answer. The measure is the model's success rate relative to each benchmark's base rate, averaged
  over benchmarks (as in G1 and R2).
- **Clustering:** 1-D tiers (quantiles of strength): simple, ordered, stable and easy to route to.
- **Number of clusters K:** small (2–4), so every specialist keeps enough training rows. Chosen on the tuning draws;
  K = 1 is the v3 prior and serves as a sanity check.
- **No leakage:** strengths, clusters and specialists are recomputed in every evaluation cell, from that cell's training
  data only.

## 3. Specialists

- **Prior:** the v3 recommended prior (`R1+B0+G1+I2`), trained only on the rows of the cluster's models (hard split:
  each row trains exactly one specialist), with the same model settings and weights as v3.
- **Start:** each specialist's R1 is its cluster's mean base rate over the training benchmarks, so a strong cluster
  starts higher on a hidden benchmark.
- **Test-time:** T2 on top of the routed prior, unchanged from v3.

## 4. Routing

- **Known model:** its training cluster. This only happens in the secondary regime (known model).
- **Unknown model:** a classifier gives cluster probabilities from the fields `predict()` sees: provider, size or
  tier, reasoning, effort, dates, harness. Most of these are often missing, so the router must work with what is
  there. In the primary regime every model is new, so routing is where results will be decided.
  - Feature choice: the fields that best predict the cluster, checked against the model features the v3 assessor
    relies on.
  - The router is trained and validated across held-out models, so its accuracy reflects new models.
- **One tier per subject:** the subject goes to the most likely cluster's specialist. Mixing specialists and
  re-routing with labels are future experiments.

## 5. Evaluation (protocol v3, unchanged)

- **Draws:** same folds (10 benchmark × 5 model), 5 repetitions, the same draws, splits and acquisition orders, both
  regimes. The primary regime (unseen benchmark, unseen model) decides.
- **Tuning:** K and the test-time scales are chosen on the tuning draws only.
- **Comparison:** paired against the v3 assessor and against B3, with a cluster bootstrap over cells. A result counts
  when its 95% interval excludes zero.
- **Continuous alternative: the general assessor.** One assessor for all models, with the model's performance as
  continuous inputs instead of splitting the data: its benchmark-adjusted strength and its plain mean score over all
  training benchmarks. It is the limit of many clusters without the data split, so it tells whether the gain needs
  specialists or only the model's level.
  - *Known model:* its values measured on the training data.
  - *Unknown model:* its values predicted from the subject fields (a regression version of the router).
  - A third input says which of the two it is (model known or not).
  - *Training rows* get their values the same way, by cross-validation over the training benchmarks: a row's model
    gets its values measured on the other folds if it appears there, otherwise values predicted from its fields. The
    assessor thus trains on both cases, as it meets them at test time, and learns how much to trust each.
  - Unlike R2 (missing for every unknown model), the inputs are never missing: a new model gets predicted values.
  - With labels, T2's subject correction already moves the pair's level, so no extra re-estimation is needed.
- **Assessors compared** (all with T2 on top, next to the v3 baselines B1, B2, B3 and the oracle level):

  | Name | What it is |
  |---|---|
  | v3 assessor | the v3 recommended prior `R1+B0+G1+I2`: one assessor for all models, the bar to beat |
  | tier router assessor | the router picks the subject's tier; that tier's specialist predicts |
  | tier router assessor, true tier | the same with the subject's true tier: the ceiling of routing |
  | general assessor | one assessor with the model performance inputs: measured for known models, predicted for unknown ones |
  | general assessor, true values | the same with the true values: its ceiling |

  The two "true" versions use the test subject's own results, so they are diagnostics, not submissions. The gap
  between each and its routed version is what the router loses.
- **Diagnostics:**
  - *Router accuracy* on unseen models, per regime.
  - *Cluster as a feature:* one assessor with the cluster as an extra feature instead of separate specialists. This
    shows whether the gain needs separate models or only the level information.
  - Breakdowns by budget (budget 0 vs later) and by cluster.

## 6. Steps

1. Strengths and clusters on the full training data: cluster sizes and how much of the variance in pair levels they
   explain.
2. Specialists with the true tier: if this does not beat the v3 assessor, stop here.
3. A router for unknown models: accuracy, then the tier router assessor.
4. The general assessor, true values then predicted, against the tier router assessor.
5. Variants: K, cluster as a feature.

## 7. Risks

- **Less data per specialist:** a specialist sees only its cluster's rows; small clusters overfit.
- **Overlap with G1:** G1 already gives the prior a group-level adjustment, so part of the expected gain may already be
  in the reference.
- **Thin subject fields:** release date is often missing and names are noisy, so routing new models may be close to
  guessing. Mixing specialists and re-routing with labels (future experiments) are the safeguards.
- **Fragile strengths:** models seen on few benchmarks get unreliable strengths, and so unreliable clusters.

## 8. Future experiments

Decided for now: clusters by model, 1-D tiers, hard split. To try once the basic version works:

- **Stability of the clusters:** how often a model changes cluster across repetitions and cells. Unstable clusters
  would make the specialists inconsistent.
- **Clusters by subject (model + settings):** one strength per subject, so settings that change performance (effort,
  harness) can move a subject between clusters. Limited by how often the settings are given. A middle option: the model's
  strength shifted by an adjustment for its settings, learned across all models.
- **Per-domain profiles:** strength per domain (maths, code, knowledge, safety…) and clusters on those profiles, to
  capture specialist models. Only worth trying if 1-D tiers show a gain under oracle routing.
- **Mixing specialists:** average the specialists' predictions, weighted by the router's cluster probabilities. More
  robust when the router is unsure.
- **Re-routing with labels:** once labels are revealed, update the cluster probabilities with how well each specialist
  predicted them, so misrouted subjects recover after a few labels.
- **Soft membership:** each specialist trains on all rows, weighted by how close each model's strength is to the
  cluster's centre. More data per specialist and smoother boundaries; the kernel width is tuned on the tuning draws
  (very wide = K copies of the v3 prior, very narrow = the hard split).
