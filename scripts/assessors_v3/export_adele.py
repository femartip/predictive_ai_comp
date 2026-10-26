"""Copy the ADeLe v1 annotation tables into the repository, slimmed to what the assessors use.

    python scripts/assessors_v3/export_adele.py --src /data/datasets/annotations/adele_v1 [--dst annotations/adele_v1]

The published tables (scripts/preprocess/annotations/vent/publish.sh) sit in one subfolder per conversion run of the
bank (no-assets/, asset-image/); the copy keeps that layout. They hold, per rubric, the expected level, the
argmax and the six level probabilities. The repository keeps, per item: item_id, text_id, truncated_any,
digit_mass_min and the 18 expected levels <code>_exp (about a tenth of the size). No item text is stored.
Rerun when more benchmarks are published: new or changed tables are rewritten, the others are left untouched (so git
only sees real changes). Writes <dst>/README.md and manifest.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ADELE_CODES, ROOT  # noqa: E402

KEEP = ["item_id", "text_id", "truncated_any", "digit_mass_min"] + [f"{c}_exp" for c in ADELE_CODES]

README = """# ADeLe v1.0 demand levels of the AI Measurement Data Bank, slim copy

One Parquet file per benchmark, one row per annotated item, in one subfolder per conversion run of the bank:
no-assets/ (benchmarks without assets) and asset-image/ (benchmarks with image assets; the annotator saw each image
through its written description). Columns: item_id, text_id (items with identical text share it),
truncated_any (the item was cut to the annotator's 32,767-token window), digit_mass_min (probability mass the
annotator put on the six level tokens; near 1 = it answered with a level) and, for each of the 18 rubrics (AS AT CEc
CEe CL KNa KNc KNf KNn KNs MCr MCt MCu MS QLl QLq SNs VO), <code>_exp: the expected level (sum of k * p_k, 0-5).

Produced by scripts/preprocess/annotations/ (distilled annotator ceferra/qwen2.5-7b-adele-annotator) on at most 600
items per benchmark, chosen deterministically (select_items.py); copied here by scripts/assessors_v3/export_adele.py
from the full tables on vent (/data/datasets/annotations/adele_v1, which also keep level probabilities). Read by
scripts/assessors_v3/build_data.py. No item text is stored.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="published folder: <run>/<benchmark>.parquet")
    ap.add_argument("--dst", default=str(ROOT / "annotations" / "adele_v1"))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rows, changed = [], 0
    for f in sorted(src.glob("*/*.parquet")):
        run = f.parent.name
        t = pd.read_parquet(f, columns=KEEP)
        t = t.astype({c: "float32" for c in KEEP if c.endswith("_exp") or c == "digit_mass_min"})
        out = dst / run / f.name
        out.parent.mkdir(exist_ok=True)
        (dst / f.name).unlink(missing_ok=True)                 # the earlier flat layout
        if not out.exists() or not pd.read_parquet(out).equals(t):
            tmp = out.with_name(f".{out.name}.tmp")
            t.to_parquet(tmp, index=False)
            tmp.replace(out)
            changed += 1
        rows.append({"benchmark": f.stem, "run": run, "items": len(t), "texts": int(t["text_id"].nunique()),
                     "truncated_items": int(t["truncated_any"].sum())})
    pd.DataFrame(rows).to_csv(dst / "manifest.csv", index=False)
    (dst / "README.md").write_text(README, encoding="utf-8")
    size = sum(p.stat().st_size for p in dst.glob("*/*.parquet")) / 2 ** 20
    print(f"{len(rows)} benchmarks in {dst} ({changed} new or changed), {size:.1f} MB")


if __name__ == "__main__":
    main()
