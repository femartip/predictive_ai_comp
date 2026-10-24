#!/usr/bin/env bash
# Annotate benchmarks on vent's GPU (run inside tmux; rerun the same command to resume after any stop):
#   tmux new -s adele "bash scripts/preprocess/annotations/vent/annotate.sh --benchmarks aegis afrieval"
#   tmux new -s adele "bash scripts/preprocess/annotations/vent/annotate.sh --benchmark-file my_share.txt"
#   bash scripts/preprocess/annotations/vent/annotate.sh --benchmarks aegis --dry-run     # check sizes, no model
# The benchmarks must have been selected first (select_items.py). Arguments go to annotate_vllm.py; --root and
# --model come from env.sh. The log is appended to $ADELE_HOME/logs/annotate_vllm.log.
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$P" "$ANN/annotate_vllm.py" --root "$ROOT" --model "$MODEL" "$@" 2>&1 | tee -a "$ADELE_HOME/logs/annotate_vllm.log"
exit "${PIPESTATUS[0]}"
