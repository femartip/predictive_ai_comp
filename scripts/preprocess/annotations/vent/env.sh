# Paths on vent, sourced by the scripts in this folder. Everything lives under your home: the shared /data tree is only
# read. ADELE_HOME: tools, environment, model and annotations; ROOT: the annotation root (one folder per benchmark).
ANN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"           # scripts/preprocess/annotations
ADELE_HOME=${ADELE_HOME:-$HOME/adele_tools}
ROOT=${ROOT:-$ADELE_HOME/annotations}
VENV=$ADELE_HOME/.venv-vllm
P=$VENV/bin/python
MODEL=${MODEL:-$ADELE_HOME/model_merged}
export PATH=$VENV/bin:$PATH        # vLLM's JIT kernels look for ninja on the PATH
mkdir -p "$ROOT" "$ADELE_HOME/logs"
