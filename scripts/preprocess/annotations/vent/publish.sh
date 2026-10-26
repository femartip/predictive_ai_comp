#!/usr/bin/env bash
# Publish finished ADeLe v1 annotation tables to the shared data folder on vent, one file per benchmark, in one
# subfolder per conversion run (team decision, 2026-10-25):
#   /data/datasets/annotations/adele_v1/no-assets/<benchmark>.parquet     benchmarks of the no-assets run
#   /data/datasets/annotations/adele_v1/asset-image/<benchmark>.parquet   benchmarks of the asset-image run
#   (+ manifest.csv with a `run` column, README.md)
# A benchmark goes to the subfolder of the run whose folder holds its JSONL (NOASSETS_JSONL, ASSETS_JSONL); one found in
# neither is reported and skipped. Copies left by the earlier flat layout (adele_v1/<benchmark>.parquet) are removed.
# Rerun whenever more benchmarks finish (e.g. after fetch_from_cluster.sh): new or changed tables are copied, the
# manifest is rebuilt from every published benchmark. Copies are atomic (temporary name, then rename); files keep the
# folder's group (datasets) and are group-writable. The tables hold no item text.
#   bash ~/adele_tools/code/vent/publish.sh [run root]        default: ~/adele_tools/runs/cap600
set -euo pipefail
RUN=${1:-$HOME/adele_tools/runs/cap600}
DEST=${DEST:-/data/datasets/annotations/adele_v1}
NOASSETS_JSONL=${NOASSETS_JSONL:-/data/datasets/measurement_db/no-assets/20261022T205359Z_b5a02e2a}
ASSETS_JSONL=${ASSETS_JSONL:-/data/datasets/measurement_db/asset-image/20261023T110411Z_26fccf60}
source "$(dirname "$0")/env.sh"
CAP=$(basename "$RUN" | sed -n 's/^cap\([0-9]*\)$/\1/p')
mkdir -p "$DEST"
chmod g+w "$DEST" 2>/dev/null || true
n_new=0
for f in "$RUN"/*/adele_v1.parquet; do
  [ -f "$f" ] || continue
  b=$(basename "$(dirname "$f")")
  if [ -f "$ASSETS_JSONL/$b.jsonl" ]; then sub=asset-image
  elif [ -f "$NOASSETS_JSONL/$b.jsonl" ]; then sub=no-assets
  else echo "  skipped $b: no JSONL in either run folder"; continue; fi
  D="$DEST/$sub"
  mkdir -p "$D" && chmod g+w "$D" 2>/dev/null || true
  if [ ! -f "$D/$b.parquet" ] || ! cmp -s "$f" "$D/$b.parquet"; then
    cp "$f" "$D/.$b.parquet.tmp" && mv "$D/.$b.parquet.tmp" "$D/$b.parquet"
    chmod g+w "$D/$b.parquet" 2>/dev/null || true
    cp "$(dirname "$f")/done.json" "$D/.$b.done.json" 2>/dev/null || true
    n_new=$((n_new + 1))
  fi
  rm -f "$DEST/$b.parquet" "$DEST/.$b.done.json"            # the earlier flat layout
done
"$P" - "$DEST" "$CAP" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
dest, cap = Path(sys.argv[1]), sys.argv[2]
rows = []
for f in sorted(dest.glob("*/*.parquet")):
    t = pd.read_parquet(f, columns=["text_id", "truncated_any", "digit_mass_min", "n_tokens_max"])
    dj = f.parent / f".{f.stem}.done.json"
    d = json.loads(dj.read_text()) if dj.exists() else {}
    rows.append({"benchmark": f.stem, "run": f.parent.name, "items": len(t), "texts": t["text_id"].nunique(), "rubrics": 18,
                 "truncated_texts": int(t.drop_duplicates("text_id")["truncated_any"].sum()),
                 "longest_prompt_tokens": int(t["n_tokens_max"].max()),
                 "digit_mass_min": round(float(t["digit_mass_min"].min()), 4),
                 "items_cap": int(cap) if cap else None, "finished": d.get("finished", "")})
m = pd.DataFrame(rows)
m.to_csv(dest / ".manifest.csv.tmp", index=False)
(dest / ".manifest.csv.tmp").replace(dest / "manifest.csv")
print(f"{len(m)} benchmarks published in {dest} ({int(m['items'].sum()):,} items): "
      + ", ".join(f"{k} {v}" for k, v in m["run"].value_counts().items()))
PY
cat > "$DEST/.README.md.tmp" <<'EOF'
# ADeLe v1.0 demand levels of the AI Measurement Data Bank

One Parquet file per benchmark, one row per annotated item, in one subfolder per conversion run of the bank:
`no-assets/` (benchmarks without assets) and `asset-image/` (benchmarks with image assets; the annotator sees each image
through its written description, the <image_description> tags of the item text). Produced with the distilled ADeLe annotator
(ceferra/qwen2.5-7b-adele-annotator, Qwen2.5-7B-Instruct + QLoRA) by scripts/preprocess/annotations/ in the
predictive_ai_comp repository (select_items.py, annotate_vllm.py on vent/cremaet, annotate_hf.py on the VRAIN cluster).

- Items: at most `items_cap` per benchmark (manifest.csv), all of them if fewer, chosen deterministically (the smallest
  SHA-1 of "0:<item_id>"); identical texts are annotated once and share their scores (`text_id`).
- Columns: benchmark, item_id, text_id; for each of the 18 rubrics (AS AT CEc CEe CL KNa KNc KNf KNn KNs MCr MCt MCu
  MS QLl QLq SNs VO): <code>_exp (expected level, sum of k * p_k), <code>_argmax, <code>_p0 ... <code>_p5 (level
  probabilities); per item: n_tokens_max (longest prompt over the rubrics), truncated_any (the item was cut at the
  32,767-token window: start and end kept, 75% of the budget at the end), digit_mass_min (probability mass on the six
  level tokens before renormalising; close to 1 means the annotator answered with a level).
- Caveats: the annotator was trained on prompts of at most 4,096 tokens; longer items are less validated. Benchmarks
  excluded by the team for truncation are not annotated (scripts/preprocess/annotations/shares/excluded.txt).
- manifest.csv: per benchmark, its run (subfolder), items, distinct texts, truncated texts, longest prompt, minimum digit mass, cap, date.
- No item text is stored here.
EOF
mv "$DEST/.README.md.tmp" "$DEST/README.md"
chmod g+w "$DEST"/manifest.csv "$DEST"/README.md 2>/dev/null || true
left=$(ls "$DEST"/*.parquet 2>/dev/null | wc -l)
[ "$left" -eq 0 ] || echo "note: $left table(s) still at the top level of $DEST (not in the run root?)"
echo "copied or updated: $n_new table(s)"
