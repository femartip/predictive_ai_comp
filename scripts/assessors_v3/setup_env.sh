#!/usr/bin/env bash
# Create the Python environment of the v3 assessor pipeline (vent, cremaet or any Linux machine).
#   bash scripts/assessors_v3/setup_env.sh            -> ~/assessors_v3/.venv
#   PYTHON=python3.12 VENV=/path/.venv bash scripts/assessors_v3/setup_env.sh
# The venv is made without pip and pip is bootstrapped with get-pip.py, because some servers' Pythons lack ensurepip
# (as on vent). LightGBM's PyPI wheel for Linux includes its OpenCL GPU learner; evaluate.py --device auto checks
# whether it works on this machine and falls back to the CPU otherwise.
set -euo pipefail
PYTHON=${PYTHON:-python3}
VENV=${VENV:-$HOME/assessors_v3/.venv}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$VENV")"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv --without-pip "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$VENV/get-pip.py"
  "$VENV/bin/python" "$VENV/get-pip.py" --quiet
fi
"$VENV/bin/python" -m pip install --quiet -r "$HERE/requirements.txt"
"$VENV/bin/python" - <<'PY'
import lightgbm, numpy, pandas, scipy, orjson
print("ok: lightgbm", lightgbm.__version__, "pandas", pandas.__version__)
PY
echo "environment ready: $VENV"
