"""ADeLe v1.0 demand levels (18 rubrics) with the distilled annotator, served by vLLM: the engine for vent's GPU.

    python annotate_vllm.py --root <annotation root> --model <merged model dir>
        (--benchmarks a b ... | --benchmark-file list.txt | --all) [--dry-run] [--gpu-mem 0.80] [--min-free-ram-gb 6]

Input: <root>/<benchmark>/items.jsonl from select_items.py. Output: <root>/<benchmark>/adele_v1.parquet (one row per
item: <code>_exp, <code>_argmax, <code>_p0..p5 for the 18 rubrics), written when every (text, rubric) pair has a
score; until then, resumable CSV parts. Benchmarks are processed one after another with one model load; finished ones
are skipped, and one locked by another process stops the run with a message.

Method (as for the six measurement-db benchmarks; prompt format in adele_prompt.py): one forward pass per (text,
rubric), no generation; the probabilities of the tokens "0"-"5" after the start of the trained answer, renormalised
(`digit_mass` = their raw total, close to 1). Window 32,767 tokens (vLLM counts the one generated token in Qwen2.5's
32,768); a longer item keeps its start and end, 75% of the budget at the end. vLLM's prefix caching computes each
rubric's text once.

Shared-machine safety (vent also runs other people's work):
- GPU: vLLM reserves --gpu-mem of the card (default 0.80) and never grows beyond it; before loading, the script
  checks that this much is free and refuses to start otherwise. --max-seqs (64) bounds the log-probability buffers that
  live outside the reservation.
- RAM: texts are read in chunks of --batch-chars characters and prompts built for one rubric at a time; free RAM is
  checked before every batch and the run stops cleanly (exit code 3) below --min-free-ram-gb, keeping its work.
- --dry-run reads, tokenises and batches everything without loading the model, to check inputs and sizes first.
"""

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# FlashInfer's sampler compiles at start-up with the system nvcc (too old on some servers); one token's log-probabilities
# need only vLLM's precompiled PyTorch sampler. Expandable segments reduce fragmentation on long prompts.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import annotate_common as C  # noqa: E402


def gpu_check(fraction, margin_gb):
    """Refuse to start unless `fraction` of GPU 0 plus a margin is free right now. Asks nvidia-smi, so the check
    itself creates no CUDA context (which would hold ~0.5 GB of the card for the whole run)."""
    def smi(query):
        return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    name, free, total = [x.strip() for x in smi("gpu=name,memory.free,memory.total").splitlines()[0].split(",")]
    free_gb, total_gb = float(free) / 1024, float(total) / 1024
    need = fraction * total_gb + margin_gb
    print(f"GPU {name}: {free_gb:.1f} GiB free of {total_gb:.1f}; vLLM will reserve {fraction * total_gb:.1f} GiB",
          flush=True)
    if free_gb < need:
        apps = smi("compute-apps=pid,process_name,used_memory") or "(none listed)"
        raise SystemExit(f"not enough free GPU memory: need {need:.1f} GiB, have {free_gb:.1f}. Other processes on the "
                         f"GPU (pid, name, MiB):\n{apps}\nWait for them, or lower --gpu-mem.")


def main():
    ap = argparse.ArgumentParser()
    C.add_common_args(ap)
    ap.add_argument("--model", help="merged model dir (merge_adapter.py); not needed with --dry-run")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", help="tokenizer for --dry-run without --model")
    ap.add_argument("--gpu-mem", type=float, default=0.80, help="share of the GPU vLLM reserves (fixed, never grows)")
    ap.add_argument("--gpu-margin-gb", type=float, default=1.0, help="extra free GPU memory required at start")
    ap.add_argument("--max-seqs", type=int, default=64, help="vLLM max_num_seqs (sequences per step)")
    ap.add_argument("--max-batched-tokens", type=int, default=16384, help="vLLM max_num_batched_tokens")
    ap.add_argument("--logprobs", type=int, default=20, help="top log-probabilities read at the answer position")
    ap.add_argument("--start-ram-gb", type=float, default=14.0,
                    help="free RAM required before loading the model (vLLM's start-up takes several GB before the "
                         "per-batch floor is first checked)")
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
    free = C.check_ram(a.min_free_ram_gb)
    print(f"free RAM {('n/a' if free is None else f'{free:.1f} GB')}; floor {a.min_free_ram_gb} GB", flush=True)
    rubrics = C.load_rubrics(a.rubrics)
    codes = list(rubrics)

    score_fn = None
    if a.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model or a.tokenizer)
    else:
        free = C.mem_available_gb()
        if free is not None and free < a.start_ram_gb:
            raise SystemExit(f"free RAM {free:.1f} GB is below --start-ram-gb {a.start_ram_gb} GB: not loading the model "
                             "now (vLLM's start-up needs several GB); retry when the machine is less busy")
        gpu_check(a.gpu_mem, a.gpu_margin_gb)
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len + 1, enable_prefix_caching=True,
                  enable_chunked_prefill=True, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_seqs,
                  max_num_batched_tokens=a.max_batched_tokens, max_logprobs=a.logprobs)
        tok = llm.get_tokenizer()
        digit_ids = [tok.convert_tokens_to_ids(str(k)) for k in range(6)]
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=a.logprobs, detokenize=False)

        def score_fn(rows):
            outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in rows], sp, use_tqdm=False)
            res = []
            for o in outs:
                lp = o.outputs[0].logprobs[0]
                p = [math.exp(lp[d].logprob) if d in lp else 0.0 for d in digit_ids]
                mass = sum(p)
                res.append(([x / mass if mass > 0 else float("nan") for x in p], mass))
            return res
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
