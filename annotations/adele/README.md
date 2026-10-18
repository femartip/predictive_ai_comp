# ADeLe demand annotations

Demand levels (0–5) of every item, on the ADeLe v1.0 rubrics (18 dimensions) and the ADeLe v2.0 "agentic"
rubrics (12 dimensions). No item text is stored here.

| file | items | rows | annotations |
|---|---|---|---|
| `adele_measurement_db.parquet` | the six measurement-db benchmarks | 18,632 | v1 + v2, our annotator |
| `adele_battery.parquet` | ADeLe battery v1.0 (16,108 items, 20 sub-benchmarks) | 16,108 | v1 shipped with the dataset (GPT-4o) + v2, our annotator |

## Keys

- `adele_measurement_db.parquet`: `item_id` is the standardized id of `data/processed/<benchmark>.jsonl`, so the
  file joins directly on it. `build_item_id` and `raw_item_id` are the measurement-db build's ids (revision
  `54074453`). MMDocRAG was annotated once per question; the annotation is repeated for its four test conditions
  (pure-text / multimodal × 15 / 20 quotes), hence 8,000 rows for 2,000 questions.
- `adele_battery.parquet`: `item_id` is the battery's `instance_id`; `source`, `benchmark` and `task` as in the
  dataset.

## Columns

For each dimension `D` annotated by our model:

- `D_exp`: expected level, Σ k · p(k);
- `D_argmax`: most likely level;
- `D_p0` … `D_p5`: probability of each level.

In the battery, the v1 levels are `D_level`: the integer levels released with the dataset (GPT-4o, ADeLe's
reference annotations), including `UG` (unguessability), which our annotator does not score.

v1 codes: AS, AT, CEc, CEe, CL, KNa, KNc, KNf, KNn, KNs, MCr, MCt, MCu, MS, QLl, QLq, SNs, VO.
v2 codes: MMe, MMp, MMs, MSc, MSm, PLe, PLp, PLs, SNk, SNp, SPa, SPv.

## How they were produced

- **Annotator:** the distilled ADeLe annotator `ceferra/qwen2.5-7b-adele-annotator` (Qwen2.5-7B). One forward pass per
  item and rubric; the level probabilities are the softmax over the digit tokens 0–5.
- **Prompt window:** 32,768 tokens; longer items keep their head and tail. 8,192 tokens for the battery, whose
  items are short.
- **v2 rubrics:** copied from `Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation`, branch `agentic-v2`
  (`src/adele/rubrics/data_v2/`, head `0c3d5a3`, 2026-10-11).

## Caveats

- **v2 is not validated.** The annotator was trained on the v1 rubrics only, so its v2 scores are untested.
  - SNk, SNp, SPa and SPv (sensory/motor) are not in the v2 manifest and describe signals that text items do
    not carry. Treat them as unreliable.
  - On measurement-db the other v2 dimensions are largely redundant with v1 (R² 0.72–0.86 from the v1 levels).
- **Two annotators for v1.** The battery's v1 levels come from GPT-4o, measurement-db's from our model, so the
  same level can mean slightly different things across the two files.
- **Multi-SWE-bench text changes.**
  - 14 items were annotated on their older, truncated text (12,000 characters); the current build has the
    full text.
  - One item (`elastic__logstash-13880`) is new in the current build and has no annotation.
- **Licences.** Both sources are gated. This folder holds only derived levels and ids, no item text; the battery's
  terms ask not to reveal its examples.

Exported by `pipeline/export_adele.py` in the assessor repository (not available in this repository yet).
