#!/usr/bin/env bash
# Publish finished ADeLe v1 annotation tables to the shared data folder on vent, one file per benchmark:
#   /data/datasets/annotations/adele_v1/<benchmark>.parquet   (+ manifest.csv, README.md)
# Rerun whenever more benchmarks finish (e.g. after fetch_from_cluster.sh): new or changed tables are copied, the
# manifest is rebuilt from every published benchmark. Copies are atomic (temporary name, then rename); files keep the
# folder's group (datasets) and are group-writable. The tables hold no item text.
#   bash ~/adele_tools/code/vent/publish.sh [run root]        default: ~/adele_tools/runs/cap600
set -euo pipefail
RUN=${1:-$HOME/adele_tools/runs/cap600}
DEST=${DEST:-/data/datasets/annotations/adele_v1}
source "$(dirname "$0")/env.sh"
CAP=$(basename "$RUN" | sed -n 's/^cap\([0-9]*\)$/\1/p')
mkdir -p "$DEST"
chmod g+w "$DEST" 2>/dev/null || true
n_new=0
for f in "$RUN"/*/adele_v1.parquet; do
  [ -f "$f" ] || continue
  b=$(basename "$(dirname "$f")")
  if [ ! -f "$DEST/$b.parquet" ] || ! cmp -s "$f" "$DEST/$b.parquet"; then
    cp "$f" "$DEST/.$b.parquet.tmp" && mv "$DEST/.$b.parquet.tmp" "$DEST/$b.parquet"
    chmod g+w "$DEST/$b.parquet" 2>/dev/null || true
    cp "$(dirname "$f")/done.json" "$DEST/.$b.done.json" 2>/dev/null || true
    n_new=$((n_new + 1))
  fi
done
"$P" - "$DEST" "$CAP" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
dest, cap = Path(sys.argv[1]), sys.argv[2]
rows = []
for f in sorted(dest.glob("*.parquet")):
    t = pd.read_parquet(f, columns=["text_id", "truncated_any", "digit_mass_min", "n_tokens_max"])
    d = json.loads((dest / f".{f.stem}.done.json").read_text()) if (dest / f".{f.stem}.done.json").exists() else {}
    rows.append({"benchmark": f.stem, "items": len(t), "texts": t["text_id"].nunique(), "rubrics": 18,
                 "truncated_texts": int(t.drop_duplicates("text_id")["truncated_any"].sum()),
                 "longest_prompt_tokens": int(t["n_tokens_max"].max()),
                 "digit_mass_min": round(float(t["digit_mass_min"].min()), 4),
                 "items_cap": int(cap) if cap else None, "finished": d.get("finished", "")})
m = pd.DataFrame(rows)
m.to_csv(dest / ".manifest.csv.tmp", index=False)
(dest / ".manifest.csv.tmp").replace(dest / "manifest.csv")
print(f"{len(m)} benchmarks published in {dest} ({int(m['items'].sum()):,} items)")
PY
cat > "$DEST/.README.md.tmp" <<'EOF'
# ADeLe v1.0 demand levels of the AI Measurement Data Bank (no-assets run)

One Parquet file per benchmark, one row per annotated item. Produced with the distilled ADeLe annotator
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
- manifest.csv: per benchmark, items, distinct texts, truncated texts, longest prompt, minimum digit mass, cap, date.
- No item text is stored here.
EOF
mv "$DEST/.README.md.tmp" "$DEST/README.md"
chmod g+w "$DEST"/manifest.csv "$DEST"/README.md 2>/dev/null || true
echo "copied or updated: $n_new table(s)"
