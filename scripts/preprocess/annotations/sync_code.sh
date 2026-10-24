#!/usr/bin/env bash
# From the laptop, repo root (predictive_ai_comp): copy the annotation code (scripts and the 18 rubrics; never data) to
#   vent:     ~/adele_tools/code                                   bash scripts/preprocess/annotations/sync_code.sh vent
#   cluster:  /data/<upv-user>@upvnet.upv.es/adele_bank/code       bash scripts/preprocess/annotations/sync_code.sh cluster <upv-user>
# SSH and SCP can point to other clients (e.g. SSH=/c/Windows/System32/OpenSSH/ssh.exe). The vent target uses the
# ssh host "vent" (an alias in ~/.ssh/config, or VENT_HOST=<user>@vent.malv.upv.es).
set -euo pipefail
SSH=${SSH:-ssh}; SCP=${SCP:-scp}
D=scripts/preprocess/annotations
case "${1:-}" in
  vent)    H=${VENT_HOST:-vent}; R='adele_tools/code' ;;
  cluster) [ -n "${2:-}" ] || { echo "usage: $0 cluster <upv-user>" >&2; exit 1; }
           H="$2@upvnet.upv.es@vrhpcadm1.dsic.upv.es"; R="/data/$2@upvnet.upv.es/adele_bank/code" ;;
  *)       echo "usage: $0 vent | cluster <upv-user>" >&2; exit 1 ;;
esac
$SSH "$H" "mkdir -p $R/rubrics/adele_v1 $R/vent $R/vrain $R/shares"
$SCP -q $D/adele_prompt.py $D/annotate_common.py $D/annotate_vllm.py $D/annotate_hf.py $D/select_items.py \
    $D/merge_adapter.py $D/compare.py $D/finalize.py $D/README.md "$H:$R/"
$SCP -q $D/rubrics/adele_v1/*.txt "$H:$R/rubrics/adele_v1/"
$SCP -q $D/vent/*.sh "$H:$R/vent/"
$SCP -q $D/vrain/* "$H:$R/vrain/"
$SCP -q $D/shares/*.txt "$H:$R/shares/"
$SSH "$H" "cd $R && sed -i 's/\r\$//' *.py vent/*.sh vrain/* shares/*.txt rubrics/adele_v1/*.txt && echo synced to $H:$R"
