# ADeLe v1.0 annotation of the data bank

Annotates bank items on the **18 ADeLe v1.0 rubrics** with the distilled annotator
[`ceferra/qwen2.5-7b-adele-annotator`](https://huggingface.co/ceferra/qwen2.5-7b-adele-annotator) (Qwen2.5-7B-Instruct +
QLoRA), one benchmark at a time, so the work can be split between people and machines:

- **vent** (one RTX 5090, 32 GB): `annotate_vllm.py`, vLLM;
- **the VRAIN Slurm cluster** (A40s, 48 GB): `annotate_hf.py`, plain transformers, one benchmark per array task.

Both read the same inputs, build the same prompts (`adele_prompt.py`) and write the same outputs. They agree to
within numeric noise: on 159 items annotated by both, mean |Δ expected level| 0.022, same most likely level 98%.

## Method

- **Prompt:** the annotator's training format: the rubric, "TASK INSTANCE: <item text>", the instruction, and the start
  of the trained answer ("… the criteria described for level "). Rubrics in `rubrics/adele_v1/` (delean-batch-manager,
  the adapter's training source).
- **Score:** one forward pass per (text, rubric), no generation; the probabilities of the tokens "0"–"5" at the next
  position, renormalised. `digit_mass`, their raw total, should be close to 1.
- **Window:** 32,767 tokens on both engines. A longer item keeps its start and end, 75% of the budget at the end. In
  the no-assets run, 0.27% of items exceed it (see `scripts/analysis/token_lengths.py` and the main README).
- **Output per item** (`<root>/<benchmark>/adele_v1.parquet`):
  - per rubric: `<code>_exp` (Σ k·p_k), `<code>_argmax` and `<code>_p0` … `<code>_p5`;
  - per item: `n_tokens_max`, `truncated_any` and `digit_mass_min`.

## Steps

**1. Select the items** (vent, CPU, where the converter's JSONL files are):

```bash
bash vent/select.sh --jsonl-dir /data/datasets/measurement_db/no-assets/<RUN_ID> \
     --benchmarks aegis afrieval --max-items 600
```

- **Selection:** deterministic, so identical on every machine. An item is chosen if SHA-1("seed:item_id") is among the
  `--max-items` smallest of its benchmark. `--all-items` takes every item; `--item-list items.csv` takes exactly the
  listed `benchmark,item_id` pairs.
- **Output:** a small `<root>/<benchmark>/` folder, with `items.jsonl` holding one copy of each distinct text, plus
  `item_map.csv` and `selection.json`. Items with empty text are skipped and counted.
- **Memory:** bounded whatever the file size. It streams the JSONL twice and holds only item IDs, file offsets and the
  chosen set (the largest no-assets file is 146 GB). It never writes inside the data folder.

**2a. Annotate on vent** (inside tmux; rerun the same command to resume):

```bash
bash vent/setup.sh                                                   # once: vLLM venv + merged model (needs ~16 GB GPU free)
bash vent/annotate.sh --benchmarks aegis afrieval --dry-run          # sizes only, no model
tmux new -s adele "bash vent/annotate.sh --benchmarks aegis afrieval"
```

**2b. Annotate on the cluster** (one benchmark per array task):

```bash
# on vent: copy the selected benchmarks to the cluster (asks for the UPVNET password)
scp -r ~/adele_tools/annotations/aegis <upv-user>@upvnet.upv.es@vrhpcadm1.dsic.upv.es:/data/<upv-user>@upvnet.upv.es/adele_bank/annotations/
# on the cluster login node, in /data/$USER/adele_bank
printf 'aegis\nafrieval\n' > my_share.txt
bash code/vrain/preflight.sh my_share.txt
n=$(grep -cv '^\s*\(#\|$\)' my_share.txt); sbatch --array=0-$((n-1))%8 --export=ALL,LIST=my_share.txt code/vrain/annotate_array.slurm
```

`%8` caps the GPUs used at once. A resubmitted task continues from its saved parts; finished benchmarks are skipped.

**3. Collect.** Each finished benchmark has its own `adele_v1.parquet`, which needs no further step. Copy the cluster's
finished folders back with `scp -r`.

**Asset benchmarks** (the asset-image conversion run): the same pipeline, with `JSONL_DIR` pointing at that run.
Images reach the annotator as their written descriptions (`<image_description>` tags in the item text). On vent, with
the list `shares/cluster_assets.txt` (the team's exclusions are in `shares/excluded.txt`):

```bash
JSONL_DIR=/data/datasets/measurement_db/asset-image/20261023T110411Z_26fccf60 \
  bash ~/adele_tools/code/vent/send_to_cluster.sh <upv-user> ~/adele_tools/code/shares/cluster_assets.txt 600
# then the printed cluster commands; when the tasks finish, on vent:
bash ~/adele_tools/code/vent/fetch_from_cluster.sh <upv-user> ~/adele_tools/code/shares/cluster_assets.txt 600
bash ~/adele_tools/code/vent/publish.sh
```

**Code onto the machines** (from the laptop, repo root):

```bash
bash scripts/preprocess/annotations/sync_code.sh vent                  # -> ~/adele_tools/code
bash scripts/preprocess/annotations/sync_code.sh cluster <upv-user>    # -> /data/<upv-user>@upvnet.upv.es/adele_bank/code
```

## Safety on a shared machine

| guard | what it does |
|---|---|
| GPU reservation | vLLM reserves `--gpu-mem` of the card (vent default 0.80 = 25.6 of 32 GB) and never grows beyond it. Before loading, `nvidia-smi` is asked whether that much is free; if not, the script refuses and lists the GPU's processes. The check itself creates no CUDA context. |
| bounded batches | Texts are read in chunks of `--batch-chars` (20M) characters and prompts built for one rubric at a time, so the prompts held in memory never exceed one chunk × one rubric. vLLM steps hold at most 64 sequences. |
| RAM floor | Free RAM is checked before every batch. Below `--min-free-ram-gb` (6 GB) the run stops cleanly with exit code 3, keeping everything written; rerun to resume. |
| locks | `<benchmark>/LOCK` is created atomically, so a benchmark is worked on by one process at a time; another run skips it and carries on with the rest. A lock left by a crashed run is removed with `--steal-lock`, after checking that the process is gone. |
| atomic writes | Parts, selections and final tables are written to temporary names and renamed: a crash never leaves a half-written file. |
| dry run | `--dry-run` reads, tokenises and batches everything without loading the model: a check of inputs and sizes before using the GPU. |

## Files

| file | what it does |
|---|---|
| `adele_prompt.py` | rubrics, prompt format, token IDs per (text, rubric) |
| `select_items.py` | step 1: item selection from the converter's JSONL |
| `annotate_common.py` | what both engines share: batching, guards, locks, parts, final table |
| `annotate_vllm.py` | step 2 on vent (vLLM) |
| `annotate_hf.py` | step 2 on the cluster (transformers; the engine that annotated the six) |
| `merge_adapter.py` | adapter + base model → one merged model (used by `vent/setup.sh`) |
| `compare.py` | agreement of annotations with a reference table |
| `vent/` | `env.sh` (paths under `~/adele_tools`), `setup.sh`, `select.sh`, `annotate.sh` |
| `vrain/` | `preflight.sh` (read-only checks), `annotate_array.slurm` (one benchmark per task) |
| `sync_code.sh` | copies the code to vent or the cluster |

## Caveats

- **The annotator was trained on prompts of at most 4,096 tokens,** on the ADeLe battery's tasks. 6% of the bank's
  items are longer than that, so their scores are less tested.
- **Text only.** Items whose stimulus is only an asset are skipped at selection. Asset descriptions appended by the
  converter are annotated as text.
- **Throughput** measured on 2026-10-21, pilot sample: cremaet's A5000 with vLLM did 4.2 s per item (18 rubrics), and
  one cluster A40 with transformers about 10.4 s. The 5090 on vent is untested.
