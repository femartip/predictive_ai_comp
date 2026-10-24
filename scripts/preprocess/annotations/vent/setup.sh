#!/usr/bin/env bash
# One-off setup on vent (run by hand; skips what already exists):
#  1. a venv for vLLM in $ADELE_HOME/.venv-vllm. vent's Python lacks the python3-venv package (needs sudo), so the
#     venv is created without pip and pip is bootstrapped inside it; nothing system-wide is touched;
#  2. the merged annotator (base Qwen2.5-7B-Instruct + the distilled ADeLe adapter) in $ADELE_HOME/model_merged.
#     Merging loads the base onto the GPU: it needs ~16 GB of it free, and refuses to start otherwise.
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ ! -x "$P" ]; then
  python3 -m venv --without-pip "$VENV"
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$ADELE_HOME/get-pip.py"
  "$P" "$ADELE_HOME/get-pip.py" -q
  rm -f "$ADELE_HOME/get-pip.py"
fi
"$P" -m pip install -q vllm peft pandas pyarrow orjson ninja
"$P" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
if [ ! -f "$MODEL/config.json" ]; then
  ram=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ "$ram" -lt 12 ]; then
    echo "only ${ram} GB of RAM free; merging (and saving 15 GB of weights) wants at least 12 GB: retry later"
    exit 1
  fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -lt 17000 ]; then
    echo "only ${free} MiB of GPU memory free; merging needs ~16 GB. Other GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
  fi
  "$P" "$ANN/merge_adapter.py" --out "$MODEL"
fi
echo "SETUP DONE"
