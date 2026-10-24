#!/bin/bash
# READ-ONLY checks on the login node before submitting:  bash /data/$USER/adele_bank/code/vrain/preflight.sh [LIST]
# Creates, modifies and deletes nothing; no GPU or heavy work. Every line prints OK, WARN or FAIL.
set -u
T1_RUN=/data/$USER/dist_delean_run
RUN=/data/$USER/adele_bank
LIST=${1:-}
fails=0
ok()   { echo "  OK    $*"; }
warn() { echo "  WARN  $*"; }
fail() { echo "  FAIL  $*"; fails=$((fails + 1)); }
check_file() { if [ -f "$1" ]; then ok "$2"; else fail "$2 (missing: $1)"; fi; }

echo "== where am I"
echo "  host $(hostname), user $USER"
case "$(hostname)" in vrhpcadm1*) ok "on the login node";; *) warn "not on vrhpcadm1";; esac

echo "== T1 environment and merged model (read only)"
check_file "$T1_RUN/conda_sh_path.txt" "conda pointer file"
if [ -f "$T1_RUN/conda_sh_path.txt" ]; then
    CONDA_SH="$(cat "$T1_RUN/conda_sh_path.txt")"
    check_file "$CONDA_SH" "conda.sh it points to"
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        if conda env list 2>/dev/null | grep -q "^distill-ft "; then
            ok "conda env distill-ft exists"
            ENV_PY="$(conda env list | awk '$1=="distill-ft"{print $NF}')/bin/python"
            "$ENV_PY" -c 'import importlib.metadata as m
for pkg in ("torch", "transformers", "pandas", "pyarrow"):
    try:
        print(f"  OK    {pkg} {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"  FAIL  {pkg} not installed in distill-ft")' 2>/dev/null || warn "could not read package versions"
        else
            fail "conda env distill-ft not found (T1: scripts/cluster/00_setup_env.sh)"
        fi
    fi
fi
M="$T1_RUN/distill_merged"
check_file "$M/config.json" "merged model config (T1: scripts/cluster/01_merge_lora.slurm)"
n_st=$(ls "$M"/*.safetensors 2>/dev/null | wc -l)
if [ "$n_st" -gt 0 ]; then ok "merged model weights: $n_st safetensors file(s), $(du -sh "$M" 2>/dev/null | cut -f1)"
else fail "no .safetensors in $M"; fi

echo "== code"
for f in adele_prompt.py annotate_common.py annotate_hf.py vrain/annotate_array.slurm; do check_file "$RUN/code/$f" "code/$f"; done
n_rub=$(ls "$RUN"/code/rubrics/adele_v1/*.txt 2>/dev/null | wc -l)
if [ "$n_rub" -eq 18 ]; then ok "18 v1 rubric files"; else fail "expected 18 rubric files in code/rubrics/adele_v1/, found $n_rub"; fi
crlf=$(grep -l $'\r' "$RUN"/code/*.py "$RUN"/code/vrain/* "$RUN"/code/rubrics/adele_v1/*.txt 2>/dev/null)
if [ -z "$crlf" ]; then ok "no Windows line endings"; else fail "Windows line endings in: $crlf"; fi
if [ -d "$RUN/logs" ]; then ok "logs/ exists"; else fail "logs/ missing: Slurm cannot write job output (mkdir -p $RUN/logs)"; fi

echo "== benchmarks"
n_sel=$(ls "$RUN"/annotations/*/items.jsonl 2>/dev/null | wc -l)
n_done=$(ls "$RUN"/annotations/*/adele_v1.parquet 2>/dev/null | wc -l)
n_lock=$(ls "$RUN"/annotations/*/LOCK 2>/dev/null | wc -l)
echo "  selected: $n_sel, finished: $n_done, locked (in progress or crashed): $n_lock"
if [ -n "$LIST" ]; then
    if [ -f "$RUN/$LIST" ]; then
        while read -r b; do
            case "$b" in ''|\#*) continue;; esac
            if [ -f "$RUN/annotations/$b/adele_v1.parquet" ]; then ok "$b: already finished (will be skipped)"
            elif [ -f "$RUN/annotations/$b/items.jsonl" ]; then ok "$b: selected, $(wc -l < "$RUN/annotations/$b/items.jsonl") texts"
            else fail "$b: no items.jsonl (copy its folder from vent)"; fi
        done < "$RUN/$LIST"
    else fail "list $RUN/$LIST not found"; fi
fi

echo "== Slurm (queries only)"
sinfo -p batch -o "  %P  nodes=%D  state=%t  gres=%G  timelimit=%l" 2>/dev/null | head -6
echo "  your running/pending jobs:"; squeue -u "$USER" -o "  %i %j %T %M %R" 2>/dev/null | head -10

echo "== result"
if [ "$fails" -eq 0 ]; then echo "  all checks passed"; else echo "  $fails check(s) failed: fix before submitting"; fi
