# predictive_ai_comp


## Setup

### 1. Install Poetry

This project uses [Poetry](https://python-poetry.org/) for dependencies and the virtual environment.
If you don't have it yet:

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

### 2. Install the project

From the repository root:

```bash
poetry install
```

### 3. Data

The dataset is **gated**. Before downloading:

1. Sign in on Hugging Face and request access on the
   [dataset page](https://huggingface.co/datasets/aims-foundations/measurement-db). Requests are reviewed
   manually, so wait for approval.
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Log in with the `hf` CLI (installed as a project dependency):

   ```bash
   poetry run hf auth login
   poetry run hf auth whoami   # should print your username
   ```

### 4. Run preprocessing (automatic downloads)

```bash
poetry run python scripts/preprocess/measurement_db.py
```

Category membership is saved in `scripts/preprocess/measurement_db_categories.json`. Normal runs read this
local catalog and use its pinned dataset revision; they do not scan remote benchmarks to classify them.
With no category flags, every benchmark in the catalog runs. To run only selected portions:

```bash
# Benchmarks that reference no assets
poetry run python scripts/preprocess/measurement_db.py --no-assets

# Benchmarks with image assets
poetry run python scripts/preprocess/measurement_db.py --asset-image

# Benchmarks with audio OR video assets
poetry run python scripts/preprocess/measurement_db.py --asset-audio --asset-video
```

Available flags are `--no-assets`, `--asset-image`, `--asset-audio`, `--asset-model`, `--asset-text`,
`--asset-video`, and `--asset-application`. Categories follow the source MIME major type: `model/obj`
belongs to `asset-model`, even though its handler renders images; `application/pdf`, JSON, SQL, Parquet,
SQLite, NumPy, and archives belong to `asset-application`. CSV, Markdown, plain text, Python, and shell
scripts belong to `asset-text`.

Flags combine with OR. A benchmark with multiple asset types runs if any selected category matches, and
all its items and referenced assets are processed. For example, selecting audio also processes video
when a matched benchmark contains both. A benchmark is listed once even when several flags match.
The collection manifest records the selected categories and the number of benchmarks excluded.

`alignment_faking` and `annotating_errors_wcf` are explicitly excluded from all selections, including
the default full run. Their reasons are recorded in the catalog's `excluded_benchmarks` mapping and
in each run manifest. Refreshing the catalog preserves exclusions. There are 279 eligible benchmarks
at the current pinned revision (213 without assets).

Each execution gets a separate folder, grouped by its selected categories:

```text
data/processed/measurement_db/
  no-assets/
    <UTC-timestamp>_<unique-id>/
      abstract_reason.jsonl
      ...
      measurement_db.manifest.json
      measurement_db_categories.json
      README.txt
  asset-application/
    <UTC-timestamp>_<unique-id>/
      ...
  asset-audio+asset-video/
    <UTC-timestamp>_<unique-id>/
      ...
  all/
    <UTC-timestamp>_<unique-id>/
      ...
```

Repeated runs create new folders. Category combinations use a sorted, consistent name; no category flags
(or explicitly selecting all categories) use `all`. A mixed benchmark is written once inside the current
run, and its actual categories appear in the run manifest. Separate runs remain independent even when
both include the same benchmark. Use `--output-root PATH` to change the parent of these category folders.

Each run folder contains its own JSONLs, one compact `measurement_db.manifest.json`, a category-catalog
snapshot, and a README. There are no per-benchmark manifest files. The run manifest stores shared source
metadata once and lists only each benchmark's name, filename, row count, and categories.
Any exported binary files are placed under its `assets/` folder, so the entire run can be shared as one
archive. Model downloads and the reusable asset-description cache remain outside the run folder.
The script prints the run path at startup and a ready-to-copy `tar` command when finished. For example:

```bash
tar -czf no-assets.tar.gz -C data/processed/measurement_db/no-assets RUN_ID
```

Replace `RUN_ID` with the folder name printed by the script. The run manifest is checkpointed after each
benchmark and records `running`, `complete`, `failed`, or `interrupted`, along with completed outputs and
any skipped or failed benchmark. Previously generated files in the flat `data/processed/` directory are
left in place; this layout applies to new executions.

To continue an interrupted run in the same folder:

```bash
poetry run python scripts/preprocess/measurement_db.py \
  --resume data/processed/measurement_db/no-assets/RUN_ID
```

Replace `RUN_ID` with the existing run folder's name. Resume uses that folder's saved category selection,
catalog, and dataset revision. Completed JSONLs and recorded skips are retained; the unfinished benchmark
starts again from its beginning, and missing or unexpectedly empty completed outputs are rebuilt.
Progress is appended to the existing `run.log`. A completed run with all its outputs present requires no
further benchmark processing.
The same folder cannot be processed by two instances at once. With `--resume`, omit category flags,
`--category-file`, `--output-root`, and `--refresh-categories`; memory and batch settings can still be changed.

Progress, processing stages, and full Python tracebacks are printed to the console and saved in the run's
`run.log`. The manifest records the current benchmark before processing starts, so it remains identifiable
even if the process is terminated externally. A terminal message of `Killed` indicates external termination
(for example, an OS memory kill); Python cannot catch SIGKILL or generate a traceback for it.

Output batches default to 256 response rows to bound the amount of question text held at once. DuckDB
uses four worker threads and a 4 GB working-memory budget, leaving room for Python/Arrow allocations.
These can be adjusted with `--threads` and `--memory-limit`; the budget does not cap the total process RAM.
For benchmarks without referenced assets, question text streams directly from Parquet rather than being
loaded into pandas in full. Run counts are calculated before attaching question text, keeping
large text columns out of the window calculation.
If a benchmark still uses too much memory, reduce the batch size:

```bash
poetry run python scripts/preprocess/measurement_db.py --no-assets --batch-size 64
```

The batch size changes memory use, not which responses are included. Rows remain separate runs.

To explicitly rebuild the catalog from the latest dataset revision, run:

```bash
poetry run python scripts/preprocess/measurement_db.py --refresh-categories
```

The refresh reads only item-manifest columns, never response tables or binary asset payloads, and exits
after writing the catalog. It records source MIME types and item counts for each benchmark, along with
the source revision and refresh timestamp. Use `--category-file PATH` to read or refresh another catalog.
A failed refresh leaves the existing catalog intact.

On every preprocessing run, the script:

1. Loads the dataset revision and benchmark names from the local category catalog.
2. Selects benchmarks matching any requested category, or selects all benchmarks when no flags are given.
3. Downloads the standardized benchmark, item, subject, response, and metadata tables into
   the Hugging Face cache, reusing cached files when available. It downloads a benchmark's `assets.parquet`
   only after its item manifests show that assets are referenced and every declared MIME type is supported.
   Model traces are not downloaded, read, joined, or exported, including in the asset-trial script.
4. Processes referenced assets by exact MIME subtype and appends either their verbatim content or a local
   multimodal-model description to the corresponding question. Descriptions are cached; unsupported types
   fail explicitly, and genuinely unknown binary payloads are stored as content-addressed artifacts.
5. Writes one row per released response to `<run-folder>/<benchmark>.jsonl`, preserving `response_id`,
   `trial`, `test_condition`, and `interactors`.
6. Normalizes graded scores to `[0, 1]`, where 1 is better, while retaining `raw_score` and the effective
   item-level `response_scale` and a `normalization_policy` containing only the policy name
   (for example, `"linear_min_max"` or `"logistic"`). Fully unbounded numeric scales use `sigmoid(s * raw_score)`, with
   `s = 1` for higher-is-better and `s = -1` for lower-is-better. This fixed transform maps zero to 0.5,
   uses temperature 1, and is independent of batching or the subjects present. It preserves score order
   but does not imply calibrated probabilities or comparable difficulty across benchmarks; extreme scores
   can saturate numerically at 0 or 1. Unordered or directionless scales are reported as skipped.
7. Updates the single `<run-folder>/measurement_db.manifest.json` with the shared source repository,
   revision and run timestamps, selected categories, status, compact benchmark entries, and any skips or
   failures. It creates no per-benchmark sidecars.

For a small end-to-end check of asset conversion, run:

```bash
poetry run python scripts/preprocess/measurement_db_asset_trial.py
```

This uses 20 fixed source items covering all 26 supported MIME types, reads only their selected Parquet rows,
and writes one deterministic released response per item with a supported scale under
`data/processed/asset_trial/`. Benchmarks with unsupported scales are recorded as skipped in the trial
manifest, which reports the items and MIME types actually written separately from the selected coverage.
The trial also writes only this one manifest, with no per-benchmark sidecars. The trial
intentionally filters each source manifest to the fixed assets listed in the script, so asset-heavy items do
not pull hundreds of unrelated files into the check. An uncached first run still downloads the configured local
image, audio, and video model weights needed by those samples.


### 5. Build the model table 

```bash
poetry run python scripts/preprocess/epoch_models_csv.py            # add --offline to reuse cached downloads
```

Writes `data/processed/models.csv`, one row per `normalized_name` in the benchmark files, with model attributes (parameters, architecture, context window, knowledge cutoff, thinking mode…) from Epoch AI, models.dev, ReleasedModels and the Hugging Face Hub; each value's origin is in a `*_source` column. Join it to any `<benchmark>.jsonl` on `normalized_name`.


## Analysis

Conclusions from [`scripts/analysis/eda.ipynb`](scripts/analysis/eda.ipynb):

In general: 
- Data is good with no duplicate items. 
- But unique item ID does not mean that the item is unique.
- Cross-benchmark comparison of models will be difficult as not the same models are used for each benchmark, and harness and reasoning changes throughout. 
- Coverage uneven. 

matharena:
- Really low coverage.
- Mostly text, only 20% of items need images. 
- Mix of agentic and non-agentic (multi-turn & single-turn).
- Lots of missing values on the subjets data that will need to be handled (provider, reasoning effort, norm model name). 
- No missing item data
- Evaluation: mix of exact matcher and judge. However some judge are not binary.
- Resuses problems accross competition splits.
- Models only see competitions held after their relseas, so not same items. 

mmdocrag:
- Non agentic, single-turn.
- Subject data really complete. In general, no missing data. 
- All items contain images
- Has missing scores, and a high number of responses.
- Evaluation: Judge, not binary but continuos
- Evaluate both multimodal and text-only models. 

multi-swe-bench
- Text only and multi-turn, agentic.
- Low coverage
- Minor missing data on subjects.
- Some models are runned on several harnesses. harness naming is inconsistent.
- Performance varies accross programming language and harness.


real_webagents
- Multi-turn agentic with tools 
- Subject data mostly complete with missing values just on release date. 
- Evaluation: exact match or judge

researchcodebench
- Single-turn
- Complete subject data, no missing values. 
- No traces!
- But, five trials per subject-item pair. 
- Evaluation: binary, based on unit-tests.

swe_rebench
- Agentic and multi-turn.
- Just one system 
- High number of attempts per item.

Per-benchmark notebooks, with the evidence behind each preprocessing script's decisions:
[`swe_rebench.ipynb`](scripts/analysis/swe_rebench.ipynb), [`mmdocrag.ipynb`](scripts/analysis/mmdocrag.ipynb),
[`matharena.ipynb`](scripts/analysis/matharena.ipynb), [`multi_swebench.ipynb`](scripts/analysis/multi_swebench.ipynb).

matharena (from `matharena.ipynb`):
- 1,755 items but 680 problems: proof problems are split into grading criteria, and final-answer problems
  appear under up to 7 prompt wordings (one per model configuration). `raw_item_id` is not a key.
- 109 problem texts recur across competition releases (e.g. `aime_2025` / `aime_2025_I`): split by `content_hash`.
- 52 of 104 subjects have no normalized name, provider or release date; they are 79 base models once filled.
- 138 attempts are empty solutions graded 0 (API failures, mostly one Claude Opus 4.7 batch): dropped.

multi-swe-bench (from `multi_swebench.ipynb`):
- The build keeps only resolved/unresolved instances and silently drops a quarter of the submissions: 11,692
  empty patches, 8,667 non-empty patches without a test report, 96 unavailable patches.
- Empty patches are model failures and enter as 0 (resolve rate 0.148 → 0.123); the rest stay out because
  the results files do not tell patch failures from environment failures.
- 82 subjects = 20 models × 11 scaffolds; harness names are harmonised (`MSWE-Agent` → `MSWE-agent`).

`scripts/preprocess/models.yaml` covers every subject of the six benchmarks except MMDocRAG's fine-tuned
models (dropped by its script). Every entry has the same fields; `provider` and `release_date` only fill
blanks in the build, and `null` means unknown or not verified.

### ADeLe prompt lengths of the bank (no-assets run)

[`scripts/analysis/token_lengths.py`](scripts/analysis/token_lengths.py) measures how long each item's prompt to the
ADeLe annotator would be: the item's tokens (Qwen2.5 tokenizer) plus the rubric's prompt overhead, which is
634–1,973 tokens depending on the rubric (counted with the longest). Run on the 189 benchmarks of the no-assets
conversion: 6.36M distinct items, a random sample of at most 5,000 per benchmark (468,616 measured). Results in
[`scripts/analysis/token_lengths/`](scripts/analysis/token_lengths/): per-benchmark table and plot.

| prompt longer than | share of all items (weighted by benchmark size) | about | benchmarks affected |
|---|---|---|---|
| 4,096 (the annotator's training length) | 6.4% | 404,000 items | 79 |
| 8,192 | 4.4% | 281,000 | 39 |
| 16,384 | 2.5% | 159,000 | 22 |
| 32,767 (the annotation window) | 0.27% | 17,000 | 11 |

- Items: median 186 tokens, 90th percentile 2,141, 99th 33,515, longest 2.2M (`mind2web`).
- The 32k window cuts few items, concentrated in a few benchmarks: `dqvis` (88% of items; all about 119k tokens),
  `software_world_models` (55%), `agentic_review_perturbation` (50%), `mind2web` (49%), `robust_reasoning_benchmark`
  (33%), `proteininvbench` (11%), and `cseo_bench` (2.3%, but of 242,000 items: the most in absolute terms).
- The annotator's training length is the bigger caveat: 6% of items are longer than any prompt it saw in training.
- 13 benchmarks have fewer than 80 distinct items (two have a single one): too small to be evaluated under the
  competition's rule.
- `swe_together` (withheld from the bank's public release) and `matharena_platform` (shares items with MathArena)
  are in this run.


## Preprocessing

- item_id (benchmark:instance_id) [data]
- benchmark [data]
- subject_id [data]
- response_id and trial: identify the exact released attempt [data]
- test_condition and interactors: complete the observation key [data]
- score: normalized to [0, 1], where 1 is better [derived]
- raw_score and response_scale: released grade and its effective item-level scale [data]
- n_runs: number of released trials for the observation cell [derived]
- score_type: binary / fraction / mixed / other declared benchmark type [bench]
- Question (contiene todo: System prompt + user prompt + image_url con markers especificos -> [system]\n<system prompt>\n\n[user]\n<user message> y <image:PATH> ) [data]
- ground_truth [data]
- eval_type: exact_match / multiple_choice / judge / unit_tests [derived from grading rule]
- domain [bench]
- modality: from the benchmark plus condition [bench + data]
- multi_turn [bench]
- programming_language: null when not code [data / bench]
- benchmark_release_date: stand-in for item_date [bench]
- provider [data]
- Model_family [curated] -> En model.yaml
- model_name [data]
- params_total_b, params_active_b [curated] -> model.yaml
- Architecture (MoE or dense)[curated] -> model.yaml
- Model_type (Instruct, base, thinking)[curated] -> model.yaml
- Thinking [curated/data] -> model.yaml o dataset
- reasoning_effort [data, mostly empty as usually not reported] 
- Open_weights [curated]
- release_date [data]
- Knowledge_cutoff [curated] -> model.yaml
- Context_window [curated] -> model.yaml
- Multimodal_input [currated] -> si modelo multimodal o no, differente del item
- agentic [bench: subject_type]
- harness, harness_version [data, mostly empty]
- Tools (still need to settle in the specific columns)
- access_date [data]
