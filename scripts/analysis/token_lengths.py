"""Token lengths of the bank's items, as the ADeLe annotator would see them: how many prompts get cut, and where.

    python scripts/analysis/token_lengths.py --jsonl-dir data/processed/measurement_db/no-assets/<RUN_ID> --out <dir>
        [--max-items 5000] [--seed 0] [--workers 12] [--tokenizer Qwen/Qwen2.5-7B-Instruct]

Input: the converter's JSONL files (scripts/preprocess/measurement_db.py), one row per response; the item text is
`question` (the bank's `content`, plus asset descriptions for benchmarks with assets). Each item is counted once, on a
uniform random sample of at most --max-items items per benchmark (0 = all): one streaming pass per file, the item id
found by string search and a row parsed only the first time its item appears; files run in parallel (--workers).

Prompt length = the item's tokens + the rubric's overhead (the training-format prompt around the item, from
adele_prompt.py; it differs per rubric, so an item can fit with one rubric and not with another). Counted against:
4,096 (the adapter's training length), 8,192, 16,384 and 32,767 (the annotation window); "cut" uses the longest
rubric, i.e. the item is cut in at least one rubric.

Writes (no item text):
  <out>/token_lengths_items.parquet       benchmark, item_id, item_tokens
  <out>/token_lengths_benchmarks.csv      per benchmark: items, sampled, token quantiles, share over each limit
  <out>/token_lengths.png                 distribution of prompt lengths with the limits marked
and prints the overall summary.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from multiprocessing import Pool

import numpy as np
import orjson
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocess" / "annotations"))
from adele_prompt import RUBRICS_V1, load_rubrics, skeletons  # noqa: E402  (the annotator's prompt format)

LIMITS = (4096, 8192, 16384, 32767)
LIMIT_NAMES = {4096: "adapter training length", 8192: "8k", 16384: "16k", 32767: "annotation window"}


ITEM_KEY = b'"item_id":'


def _item_id(line):
    """The item_id of a JSONL row by string search (no full JSON parse); None if not found."""
    i = line.find(ITEM_KEY)
    if i < 0:
        return None
    j = line.find(b'"', i + len(ITEM_KEY))
    k = line.find(b'"', j + 1)
    return line[j + 1:k] if j >= 0 and k > j else None


def items_of(path, max_items, seed):
    """(item_id, question) once per item of one JSONL: a uniform random sample of at most max_items items (reservoir
    sampling over distinct items, seeded), and the number of distinct items. One streaming pass: a row is fully parsed
    only the first time its item appears; memory holds the set of item ids and the sample's texts."""
    rng = np.random.default_rng(seed)
    seen, sample, n = set(), [], 0
    with open(path, "rb") as fh:
        for line in fh:
            iid = _item_id(line)
            if iid is None or iid in seen:
                if iid is None:                                   # unusual layout: parse to get the id
                    r = orjson.loads(line)
                    iid = str(r.get("item_id")).encode()
                    if iid in seen:
                        continue
                else:
                    continue
            seen.add(iid)
            n += 1
            if not max_items or len(sample) < max_items:
                r = orjson.loads(line)
                sample.append((r.get("item_id"), r.get("question") or ""))
            else:
                k = rng.integers(0, n)
                if k < max_items:
                    r = orjson.loads(line)
                    sample[k] = (r.get("item_id"), r.get("question") or "")
    return pd.DataFrame(sample, columns=["item_id", "question"]), n


_TOK = {}
BATCH_CHARS = 20_000_000          # texts tokenised together: bounds the memory of the encodings


def count_tokens(tok, texts):
    """Token count of each text, in batches of at most BATCH_CHARS characters, keeping only the counts (the encodings
    of 5,000 long texts at once took 13 GB on 2026-10-23)."""
    out, batch, size = [], [], 0
    for t in texts:
        if batch and size + len(t) > BATCH_CHARS:
            out += [len(e.ids) for e in tok.backend_tokenizer.encode_batch(batch, add_special_tokens=False)]
            batch, size = [], 0
        batch.append(t)
        size += len(t)
    if batch:
        out += [len(e.ids) for e in tok.backend_tokenizer.encode_batch(batch, add_special_tokens=False)]
    return np.array(out, dtype=int)


def measure(args):
    """One benchmark: sample its items, count their tokens and write parts/<benchmark>.parquet + .json at once (run in
    a worker). The tokenizer is loaded once per worker; run with HF_HUB_OFFLINE=1 once it is cached, or every load
    queries the Hub (and gets rate limited)."""
    path, max_items, seed, tokenizer, parts = args
    if tokenizer not in _TOK:
        from transformers import AutoTokenizer
        _TOK[tokenizer] = AutoTokenizer.from_pretrained(tokenizer)
    tok = _TOK[tokenizer]
    t0 = time.time()
    df, n = items_of(path, max_items, seed)
    lens = count_tokens(tok, df["question"].tolist())
    name = Path(path).stem
    info = {"benchmark": name, "items": n, "sampled": len(lens), "empty": int((df["question"].str.strip() == "").sum()),
            "seconds": round(time.time() - t0, 1)}
    pd.DataFrame({"benchmark": name, "item_id": df["item_id"], "item_tokens": lens}).to_parquet(
        Path(parts) / f"{name}.parquet", index=False)
    (Path(parts) / f"{name}.json").write_text(json.dumps(info))     # written last: marks the benchmark as done
    return info, lens


def plot(prompt_max, weights, out):
    """Share of items whose prompt is longer than x (log-log), as measured on the sample and weighted by each
    benchmark's real size, with the four limits marked and labelled below the curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter
    surface, ink, muted, grid = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"
    series = [("sampled items (at most 5,000 per benchmark)", np.ones(len(prompt_max)), "#2a78d6", "-"),
              ("weighted by benchmark size (all items)", weights, "#eb6834", (0, (5, 2)))]
    order = np.argsort(prompt_max)
    x = prompt_max[order]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    shares = {}
    for label, w, colour, style in series:
        w = w[order] / w.sum()
        surv = 1 - np.cumsum(w)                        # share strictly longer than each x
        ax.step(x, np.clip(surv, 1e-6, None) * 100, where="post", color=colour, linewidth=2, linestyle=style, label=label)
        shares[label] = [float(w[x > lim].sum()) for lim in LIMITS]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_ylim(0.01, 100)
    for i, lim in enumerate(LIMITS):
        s_samp, s_w = shares[series[0][0]][i], shares[series[1][0]][i]
        ax.axvline(lim, color=muted, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
        ax.text(lim * 1.05, 0.012, f"{lim:,}\n{s_samp:.1%} / {s_w:.1%}", color=ink, fontsize=7.5, va="bottom")
    ax.set_xlabel("Prompt length in tokens (item + longest rubric overhead)", color=ink, fontsize=10)
    ax.set_ylabel("Items with a longer prompt", color=ink, fontsize=10)
    ax.set_title("Share of bank items whose ADeLe prompt exceeds a length", color=ink, fontsize=12, loc="left")
    ax.grid(axis="y", color=grid, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=9)
    leg = ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=ink)
    fig.text(0.01, 0.01, "Dashed lines: 4,096 (the adapter's training length), 8,192, 16,384 and 32,767 (the annotation "
             "window). Labels: share of items over each, sampled / weighted.", color=muted, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=5000, help="random items per benchmark (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", help="the annotator's tokenizer")
    ap.add_argument("--workers", type=int, default=12, help="files processed in parallel")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    sk = skeletons(tok, load_rubrics(RUBRICS_V1))
    overhead = {c: len(h) + len(t) for c, (_, _, h, t) in sk.items()}
    o_min, o_max = min(overhead.values()), max(overhead.values())
    print(f"rubric overhead: {o_min:,}-{o_max:,} tokens (18 v1 rubrics)", flush=True)

    out = Path(a.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.jsonl_dir).glob("*.jsonl"), key=lambda f: -f.stat().st_size)   # biggest first
    todo = [f for f in files if not (parts / f"{f.stem}.json").exists()]                 # resumable
    print(f"{len(files)} files, {len(files) - len(todo)} already measured, {len(todo)} to go", flush=True)
    t0 = time.time()
    jobs = [(str(f), a.max_items, a.seed, a.tokenizer, str(parts)) for f in todo]
    with Pool(a.workers, maxtasksperchild=1) as pool:      # a fresh worker per file: memory goes back after each
        for k, (info, lens) in enumerate(pool.imap_unordered(measure, jobs)):
            print(f"  [{time.time() - t0:5.0f}s] {k + 1}/{len(todo)} {info['benchmark']}: {info['items']:,} items "
                  f"({info['sampled']:,} sampled, {info['seconds']:.0f}s), median "
                  f"{int(np.median(lens)) if len(lens) else 0:,} tokens, over the window "
                  f"{float((lens + o_max > 32767).mean()) if len(lens) else 0:.1%}", flush=True)
    rows, summary = [], []
    for f in files:
        info = json.loads((parts / f"{f.stem}.json").read_text())
        d = pd.read_parquet(parts / f"{f.stem}.parquet")
        lens = d["item_tokens"].to_numpy()
        p = lens + o_max
        rows.append(d)
        summary.append({**{k: info[k] for k in ("benchmark", "items", "sampled", "empty")},
                        "median": int(np.median(lens)) if len(lens) else 0,
                        "p95": int(np.quantile(lens, 0.95)) if len(lens) else 0,
                        "max": int(lens.max()) if len(lens) else 0,
                        **{f"over_{lim}": float((p > lim).mean()) if len(lens) else 0.0 for lim in LIMITS}})
    items = pd.concat(rows, ignore_index=True)
    items.to_parquet(out / "token_lengths_items.parquet", index=False)
    s = pd.DataFrame(summary).sort_values("over_32767", ascending=False)
    s.to_csv(out / "token_lengths_benchmarks.csv", index=False)
    p = items["item_tokens"].to_numpy() + o_max
    scale = s.set_index("benchmark")["items"] / s.set_index("benchmark")["sampled"].replace(0, 1)
    plot(p, items["benchmark"].map(scale).to_numpy(float), out / "token_lengths.png")

    print(f"\n{len(files)} benchmarks, {s['items'].sum():,} items, {len(items):,} measured "
          f"({int(s['empty'].sum()):,} without text)")
    q = np.quantile(items["item_tokens"], [0.5, 0.9, 0.99])
    print(f"item tokens: median {q[0]:,.0f}, p90 {q[1]:,.0f}, p99 {q[2]:,.0f}, max {items['item_tokens'].max():,}")
    for lim in LIMITS:
        over = p > lim
        weighted = np.average(s[f"over_{lim}"], weights=s["items"])
        print(f"prompts over {lim:>6,} ({LIMIT_NAMES[lim]}): {over.mean():6.2%} of sampled items, {weighted:6.2%} "
              f"weighted by benchmark size (~{(s[f'over_{lim}'] * s['items']).sum():,.0f} items), in "
              f"{int((s[f'over_{lim}'] > 0).sum())} benchmarks")
    print("\nbenchmarks with most items over the window:")
    print(s[s["over_32767"] > 0].head(15)[["benchmark", "items", "median", "p95", "max", "over_8192", "over_32767"]]
          .to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
