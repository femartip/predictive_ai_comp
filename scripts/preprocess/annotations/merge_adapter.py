"""Merge the distilled ADeLe adapter into Qwen2.5-7B-Instruct and save a standalone model for vLLM.

    python merge_adapter.py --out model_merged [--base Qwen/Qwen2.5-7B-Instruct]
                            [--adapter ceferra/qwen2.5-7b-adele-annotator]

The base loads straight onto the GPU (device_map="cuda"): cremaet has 14 GB of RAM, less than the 15 GB of bf16
weights. The tokenizer is saved from the adapter repository (the one used in training). Needs the GPU free (~16 GB).
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="ceferra/qwen2.5-7b-adele-annotator")
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
    print(f"merged {a.adapter} into {a.base} -> {a.out}")


if __name__ == "__main__":
    main()
