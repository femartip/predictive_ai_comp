#!/usr/bin/env bash
# Run on vent: bring the finished annotation tables (adele_v1.parquet + done.json) of a list of benchmarks back from the
# VRAIN cluster into $ADELE_HOME/runs/cap<N>, in one stream (one password prompt). Unfinished benchmarks are listed and
# left on the cluster; rerun later to fetch them.
#   bash ~/adele_tools/code/vent/fetch_from_cluster.sh <upv-user> [list] [items per benchmark]
set -euo pipefail
USER_UPV=${1:?usage: fetch_from_cluster.sh <upv-user> [list] [items per benchmark]}
LIST=${2:-$HOME/adele_tools/code/shares/cluster_long.txt}
CAP=${3:-600}
export ROOT=${ROOT:-$HOME/adele_tools/runs/cap$CAP}
source "$(dirname "$0")/env.sh"
C="$USER_UPV@upvnet.upv.es@vrhpcadm1.dsic.upv.es"
R="/data/$USER_UPV@upvnet.upv.es/adele_bank/annotations"
BENCH=$(grep -v '^\s*\(#\|$\)' "$LIST" | tr '\n' ' ')
ssh "$C" "cd '$R' && for b in $BENCH; do if [ -f \$b/adele_v1.parquet ]; then echo \$b/adele_v1.parquet \$b/done.json; \
          else echo \"not finished: \$b\" >&2; fi; done | xargs -r tar cf -" | tar xf - -C "$ROOT"
for b in $BENCH; do
  if [ -f "$ROOT/$b/adele_v1.parquet" ]; then echo "  fetched: $b"; fi
done
