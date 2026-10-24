#!/usr/bin/env bash
# Run on vent: select a list of benchmarks (if not done yet) and send them, with the list and the annotation code, to the
# VRAIN cluster in one stream (one password prompt). Then prints the commands to run on the cluster.
#   bash ~/adele_tools/code/vent/send_to_cluster.sh <upv-user> [list] [items per benchmark]
#   defaults: list = ~/adele_tools/code/shares/cluster_long.txt, 600 items per benchmark
# Selections go to $ADELE_HOME/runs/cap<N> (one root per cap, kept apart from test selections). On the cluster:
#   /data/<upv-user>@upvnet.upv.es/adele_bank/{code,annotations,logs}
set -euo pipefail
USER_UPV=${1:?usage: send_to_cluster.sh <upv-user> [list] [items per benchmark]}
LIST=${2:-$HOME/adele_tools/code/shares/cluster_long.txt}
CAP=${3:-600}
JSONL_DIR=${JSONL_DIR:-/data/datasets/measurement_db/no-assets/20261022T205359Z_b5a02e2a}
export ROOT=${ROOT:-$HOME/adele_tools/runs/cap$CAP}
source "$(dirname "$0")/env.sh"
C="$USER_UPV@upvnet.upv.es@vrhpcadm1.dsic.upv.es"
R="/data/$USER_UPV@upvnet.upv.es/adele_bank"
mapfile -t BENCH < <(grep -v '^\s*\(#\|$\)' "$LIST")
echo "== ${#BENCH[@]} benchmarks from $LIST, $CAP items each, root $ROOT"
"$P" "$ANN/select_items.py" --jsonl-dir "$JSONL_DIR" --root "$ROOT" --benchmarks "${BENCH[@]}" --max-items "$CAP" --workers 3
for b in "${BENCH[@]}"; do
  [ -f "$ROOT/$b/items.jsonl" ] || { echo "missing selection for $b"; exit 1; }
done
cp "$LIST" "$ROOT/$(basename "$LIST")"
echo "== sending to $C:$R (one password prompt)"
tar czf - -C "$ROOT" "${BENCH[@]}" "$(basename "$LIST")" -C "$ADELE_HOME" --exclude=__pycache__ code \
  | ssh "$C" "mkdir -p '$R/annotations' '$R/logs' && tar xzf - -C '$R/annotations' && rm -rf '$R/code' \
              && mv '$R/annotations/code' '$R/code' && echo received: \$(ls '$R/annotations' | wc -l) entries in annotations/"
cat <<EOF
== next, on the cluster login node:
   ssh $C
   cd $R
   bash code/vrain/preflight.sh annotations/$(basename "$LIST")
   sbatch --array=0-$((${#BENCH[@]} - 1))%${#BENCH[@]} --export=ALL,LIST=annotations/$(basename "$LIST") code/vrain/annotate_array.slurm
   squeue -u \$USER            # progress: tail -f logs/adele-<jobid>_<task>.out
== when tasks finish, on vent: bash $ANN/vent/fetch_from_cluster.sh $USER_UPV $LIST $CAP
EOF
