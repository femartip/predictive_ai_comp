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
poetry run python scripts/preprocess/mmdocrag.py
poetry run python scripts/preprocess/swe_rebench.py
poetry run python scripts/preprocess/real_webagents.py
poetry run python scripts/preprocess/researchcodebench.py
poetry run python scripts/preprocess/matharena.py
poetry run python scripts/preprocess/multi_swebench.py
```

On every run, each script:

1. Checks the latest revision of `aims-foundations/measurement-db` and pins that commit for the run.
2. Loads the required files, downloading missing inputs into the Hugging Face cache and reusing
   cached files when available. MMDocRAG also fetches raw answers, quote sets, and the prompt source
   archive; SWE-rebench also fetches raw trajectories for error filtering; MathArena also fetches the
   solution traces (API-failure filtering) and the images, written to `data/matharena/images/`;
   Multi-SWE-bench also fetches every run's `results.json` and predictions (empty-patch outcomes).
3. Rebuilds and overwrites `data/processed/<benchmark>.jsonl`.
4. Writes `data/processed/<benchmark>.manifest.json` with the source repository, exact revision,
   load timestamp, source table paths, output row count, and output SHA-256. MMDocRAG and SWE-rebench
   additionally record every source file used, including metadata and raw inputs.

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


## Preprocessing

- item_id (benchmark:instance_id) [data]
- benchmark [data]
- subject_id [data]
- score: mean over runs, in [0, 1] [data]
- n_runs [data]
- score_type: binary / fraction / mixed before averaging [bench]
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
