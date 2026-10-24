"""ADeLe v1.0 demand levels (18 rubrics) with the distilled annotator, with plain transformers: the engine for the VRAIN
Slurm cluster (one A40 per task; vrain/annotate_array.slurm runs one benchmark per array task).

    python annotate_hf.py --root <annotation root> --model <merged model dir>
        (--benchmarks a b ... | --benchmark-file list.txt | --all) [--dry-run]

Same inputs, prompts and outputs as annotate_vllm.py (see there and annotate_common.py), so a benchmark can be done on
either machine and the tables are interchangeable (checked on 2026-10-21: mean |Δ expected level| 0.022 between the two
engines on the same 159 items).

Engine: the one that annotated the six measurement-db benchmarks on this cluster. One forward pass per prompt, no key/
value cache; the probabilities of the tokens "0"-"5" at the last position. Length-sorted batches under a padded-token
budget; a batch that runs out of GPU memory is split in two and retried. SDPA attention, so a 32k prompt never
materialises the full attention matrix; 32k prompts need a 48 GB GPU (the job script excludes the 24 GB A30 nodes,
and the script refuses to start on a smaller GPU with a window above 16,384).
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import annotate_common as C


def batches(rows, max_tokens, max_batch):
    """Length-sorted batches (longest first) under a padded-token budget; yields lists of row indices."""
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["ids"]))
    batch, longest = [], 0
    for i in order:
        n = len(rows[i]["ids"])
        if batch and (max(longest, n) * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        yield batch


@torch.inference_mode()
def score(model, seqs, pad_id, digit_ids):
    """(probabilities of the six level tokens renormalised, their raw total) at the next position, left padding."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    # no mask for a single sequence, so SDPA can use its memory-efficient causal kernel
    hidden = model.model(input_ids=ids.cuda(), attention_mask=None if len(seqs) == 1 else mask.cuda(),
                         use_cache=False).last_hidden_state
    probs = torch.softmax(model.lm_head(hidden[:, -1, :]).float(), dim=-1)[:, digit_ids]
    mass = probs.sum(dim=-1, keepdim=True)
    return (probs / mass).cpu().tolist(), mass.squeeze(-1).cpu().tolist()


def _is_oom(err):
    msg = str(err)
    return isinstance(err, torch.cuda.OutOfMemoryError) or (
        isinstance(err, RuntimeError) and ("out of memory" in msg or "CUDA driver error" in msg))


def score_safe(model, seqs, pad_id, digit_ids):
    try:
        p, m = score(model, seqs, pad_id, digit_ids)
        return list(zip(p, m))
    except RuntimeError as err:
        if not _is_oom(err):
            raise
        torch.cuda.empty_cache()
        if len(seqs) == 1:
            raise
        mid = len(seqs) // 2
        return score_safe(model, seqs[:mid], pad_id, digit_ids) + score_safe(model, seqs[mid:], pad_id, digit_ids)


def main():
    ap = argparse.ArgumentParser()
    C.add_common_args(ap)
    ap.add_argument("--model", help="merged model dir; not needed with --dry-run")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", help="tokenizer for --dry-run without --model")
    ap.add_argument("--max-batch-tokens", type=int, default=65536, help="padded tokens per forward pass")
    ap.add_argument("--max-batch", type=int, default=64)
    a = ap.parse_args()
    if not a.dry_run and not a.model:
        ap.error("--model is required unless --dry-run")

    C.require_parquet()
    root = Path(a.root)
    names = C.benchmarks_from(a)
    todo = [b for b in names if not (root / b / C.FINAL).exists()]
    print(f"{len(names)} benchmarks requested, {len(names) - len(todo)} already finished, {len(todo)} to do", flush=True)
    if not todo:
        return
    C.check_ram(a.min_free_ram_gb)
    rubrics = C.load_rubrics(a.rubrics)
    codes = list(rubrics)

    score_fn = None
    if a.dry_run:
        tok = AutoTokenizer.from_pretrained(a.model or a.tokenizer)
    else:
        gpu = torch.cuda.get_device_properties(0)
        gib = gpu.total_memory / 2**30
        print(f"GPU {gpu.name}, {gib:.1f} GiB", flush=True)
        if a.max_len > 16384 and gib < 40:
            raise SystemExit(f"GPU has {gib:.1f} GiB; prompts of up to {a.max_len} tokens need a ~48 GB GPU (A40): "
                             "resubmit excluding this node, or pass --max-len 16384")
        if gib < 40:
            a.max_batch_tokens = min(a.max_batch_tokens, 16384)
        tok = AutoTokenizer.from_pretrained(a.model)
        # torch_dtype (older transformers ignore dtype=) and an explicit move to the GPU, as for the six
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").to("cuda").eval()
        digit_ids = [tok.convert_tokens_to_ids(str(k)) for k in range(6)]
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def score_fn(rows):
            out = [None] * len(rows)
            for idx in batches(rows, a.max_batch_tokens, a.max_batch):
                for i, r in zip(idx, score_safe(model, [rows[i]["ids"] for i in idx], pad_id, digit_ids)):
                    out[i] = r
            return out
    sk = C.skeletons(tok, rubrics)

    t0 = time.time()
    for b in todo:
        bdir = root / b
        try:
            with C.Lock(bdir, steal=a.steal_lock):
                n_p, n_t = C.run_benchmark(bdir, codes, tok, sk, a, score_fn)
                if a.dry_run:
                    print(f"[dry run] {b}: {n_p:,} prompts, {n_t / 1e6:.1f}M prompt tokens to score", flush=True)
                    continue
                missing = C.finalize(bdir, codes)
                print(f"{b}: {'finished -> ' + str(bdir / C.FINAL) if not missing else f'{missing} pairs missing'} "
                      f"({(time.time() - t0) / 60:.1f} min so far)", flush=True)
        except C.Locked as err:
            print(f"SKIPPED {b}: {err}", flush=True)
            continue
        except C.MemoryGuard as err:
            print(f"STOPPED on {b}: {err}", flush=True)
            sys.exit(3)
    print("all requested benchmarks processed", flush=True)


if __name__ == "__main__":
    main()
