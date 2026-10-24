#!/usr/bin/env bash
# Run a step of the v3 pipeline with the resource settings of each server.
#   bash scripts/assessors_v3/run.sh <vent|cremaet> build    [build_data.py args]
#   bash scripts/assessors_v3/run.sh <vent|cremaet> evaluate [evaluate.py args]
#   bash scripts/assessors_v3/run.sh <vent|cremaet> report   NAME [report.py args]
# Run it inside tmux (tmux new -s assessors), the full protocol takes hours.
#
# vent (shared: 48 cores, ~30 GB RAM, the GPU is used by others): CPU only, 16 threads, low priority (nice), refuses
#   to start below 10 GB of free memory and stops cleanly below 5 GB (rerun the same command to resume).
# cremaet (A5000 24 GB, 14 GB RAM): LightGBM on the GPU when it works there (else CPU), 8 threads, thresholds 6 / 2 GB.
#   cremaet has no copy of the bank: build on vent, then copy ~/assessors_v3_work/data to cremaet.
# Environment overrides (config.py): ASSESSOR_WORK, ASSESSOR_JSONL_DIR, ASSESSOR_ADELE_DIR, ASSESSOR_MODELS_CSV,
# ASSESSOR_TRAIN_PAIRS, ASSESSOR_START_RAM_GB, ASSESSOR_MIN_FREE_RAM_GB; VENV (default ~/assessors_v3/.venv).
set -euo pipefail
SERVER=${1:?usage: run.sh <vent|cremaet> <build|evaluate|report> [args]}
STEP=${2:?usage: run.sh <vent|cremaet> <build|evaluate|report> [args]}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${VENV:-$HOME/assessors_v3/.venv}/bin/python
[ -x "$PY" ] || { echo "no environment at $PY: run bash $HERE/setup_env.sh first"; exit 1; }
case "$SERVER" in
  vent)
    export ASSESSOR_START_RAM_GB=${ASSESSOR_START_RAM_GB:-10} ASSESSOR_MIN_FREE_RAM_GB=${ASSESSOR_MIN_FREE_RAM_GB:-5}
    THREADS=16; DEVICE=cpu; WORKERS=4; NICE="nice -n 10" ;;
  cremaet)
    export ASSESSOR_START_RAM_GB=${ASSESSOR_START_RAM_GB:-6} ASSESSOR_MIN_FREE_RAM_GB=${ASSESSOR_MIN_FREE_RAM_GB:-2}
    THREADS=8; DEVICE=auto; WORKERS=2; NICE="" ;;
  *) echo "unknown server '$SERVER' (vent or cremaet)"; exit 1 ;;
esac
# BLAS threads: one per process (the parallelism is across cells; LightGBM sets its own threads with --n-jobs)
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$STEP" in
  build)    exec $NICE "$PY" "$HERE/build_data.py" --workers "$WORKERS" "$@" ;;
  evaluate) exec $NICE "$PY" "$HERE/evaluate.py" --device "$DEVICE" --n-jobs "$THREADS" "$@" ;;
  report)   exec "$PY" "$HERE/report.py" "$@" ;;
  *) echo "unknown step '$STEP' (build, evaluate or report)"; exit 1 ;;
esac
