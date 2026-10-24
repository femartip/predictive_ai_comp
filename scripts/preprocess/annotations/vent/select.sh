#!/usr/bin/env bash
# Select the items to annotate (CPU only; reads the converter's JSONL files, writes compact items under $ROOT):
#   bash scripts/preprocess/annotations/vent/select.sh --jsonl-dir /data/datasets/measurement_db/no-assets/<RUN_ID> \
#        --benchmarks aegis afrieval --max-items 600
# Arguments go to select_items.py; --root comes from env.sh. Uses the vLLM venv (it has orjson).
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$P" "$ANN/select_items.py" --root "$ROOT" "$@" 2>&1 | tee -a "$ADELE_HOME/logs/select_items.log"
exit "${PIPESTATUS[0]}"
