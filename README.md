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

### 4. Download the data

The notebooks expect the dataset in `data/` at the repository root. `data/` is git-ignored.

**Everything (≈ 12 GB).** This includes each benchmark's `raw/` upstream source snapshots and
`reproduction_checks/` re-run pilots:

```bash
poetry run hf download aims-foundations/measurement-db --repo-type dataset --local-dir data
```


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
