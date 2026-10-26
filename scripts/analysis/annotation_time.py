"""Estimated GPU time to annotate the bank with ADeLe v1, per benchmark and item cap, from token_lengths.py's output.

    python scripts/analysis/annotation_time.py --token-dir <token_lengths output> [--caps 600 1000 2000 0]
        [--tokens-per-second 3800] [--gpus 20] [--tokenizer Qwen/Qwen2.5-7B-Instruct]

Cost model (scripts/preprocess/annotations/): every item is scored once per rubric (18 prompts); a prompt is the item's
tokens plus that rubric's overhead (adele_prompt.py), cut at the 32,767-token window. Work = prompt tokens processed
at --tokens-per-second: 3,800 is the transformers engine on one A40 of the VRAIN cluster, measured on 2026-10-23
(3,500-4,200). vLLM with prefix caching is several times faster on short prompts (cremaet's A5000: 25-36k tokens/s
under 2k tokens, ~3,900 at 32k), so the hours are an upper bound there. Tokens per item are the mean over the items
token_lengths.py sampled; a benchmark's items under a cap are min(items, cap) (cap 0 = all items).
Writes <token-dir>/annotation_time.csv (per benchmark: items, tokens per item, GPU-hours per cap) and prints totals.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for p in (Path(__file__).resolve().parents[1] / "preprocess" / "annotations", Path(__file__).resolve().parent):
    if (p / "adele_prompt.py").exists():
        sys.path.insert(0, str(p))
        break
from adele_prompt import RUBRICS_V1, load_rubrics, skeletons  # noqa: E402  (the annotator's prompt format)

WINDOW = 32767


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-dir", required=True, help="folder with token_lengths_items.parquet and *_benchmarks.csv")
    ap.add_argument("--caps", type=int, nargs="+", default=[600, 1000, 2000, 0], help="items per benchmark; 0 = all")
    ap.add_argument("--tokens-per-second", type=float, default=3800)
    ap.add_argument("--gpus", type=int, default=20, help="GPUs working in parallel, for the wall-time column")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    overhead = np.array([len(h) + len(t) for _, _, h, t in skeletons(tok, load_rubrics(RUBRICS_V1)).values()])

    d = Path(a.token_dir)
    items = pd.read_parquet(d / "token_lengths_items.parquet")
    bench = pd.read_csv(d / "token_lengths_benchmarks.csv").set_index("benchmark")
    # tokens processed for one item: its 18 prompts, each cut at the window
    per_item = np.minimum(items["item_tokens"].to_numpy()[:, None] + overhead[None, :], WINDOW).sum(axis=1)
    t = pd.DataFrame({"benchmark": items["benchmark"], "tokens": per_item}).groupby("benchmark")["tokens"].mean()
    out = pd.DataFrame({"items": bench["items"], "tokens_per_item": t}).fillna({"tokens_per_item": 0})
    for cap in a.caps:
        n = out["items"] if cap == 0 else np.minimum(out["items"], cap)
        out[f"gpu_hours_{cap or 'all'}"] = n * out["tokens_per_item"] / a.tokens_per_second / 3600
    out = out.sort_values(f"gpu_hours_{a.caps[-1] or 'all'}", ascending=False)
    out.round(2).to_csv(d / "annotation_time.csv")
    print(f"{len(out)} benchmarks; {a.tokens_per_second:,.0f} tokens/s per GPU; 18 prompts per item "
          f"(rubric overhead {overhead.min():,}-{overhead.max():,} tokens)")
    for cap in a.caps:
        col = f"gpu_hours_{cap or 'all'}"
        n = int((out["items"] if cap == 0 else np.minimum(out["items"], cap)).sum())
        h = out[col].sum()
        print(f"  {'all items' if cap == 0 else f'cap {cap:,}':>10}: {n:>10,} items, {h:8,.1f} GPU-hours = "
              f"{h / a.gpus:6.1f} h on {a.gpus} GPUs (longest benchmark {out[col].max():.1f} h)")
    print(out.head(10).round(1).to_string())
    print(f"wrote {d / 'annotation_time.csv'}")


if __name__ == "__main__":
    main()
