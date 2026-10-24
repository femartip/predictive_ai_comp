#!/bin/bash
#SBATCH --job-name=adele-v1
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --exclude=vrhpc4.dsic.upv.es,vrhpc5.dsic.upv.es,vrhpc6.dsic.upv.es
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/data/%u/adele_bank/logs/adele-%A_%a.out
#
# One benchmark per array task: task i annotates the benchmark on line i+1 of LIST (a file under $RUN), with
# annotate_hf.py, the 18 ADeLe v1 rubrics and a 32,767-token window. Resumable: resubmitting a task continues from its
# saved parts; a finished benchmark is skipped. Submit from $RUN, limiting concurrent GPUs with %N:
#   n=$(grep -cv '^\s*\(#\|$\)' my_share.txt); sbatch --array=0-$((n-1))%8 --export=ALL,LIST=my_share.txt annotate_array.slurm
# --exclude: vrhpc4-6 have A30 GPUs (24 GB); 32k-token prompts need an A40 (48 GB).
set -euo pipefail
T1_RUN=/data/$USER/dist_delean_run          # conda env + merged model from the T1 project
RUN=/data/$USER/adele_bank
: "${LIST:?pass the benchmark list: --export=ALL,LIST=<file under $RUN>}"
source "$(cat $T1_RUN/conda_sh_path.txt)"
conda activate distill-ft
export HF_HOME="$T1_RUN/.cache/huggingface" HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation on long prompts
cd $RUN
B=$(grep -v '^\s*\(#\|$\)' "$LIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
[ -n "$B" ] || { echo "no benchmark on line $((SLURM_ARRAY_TASK_ID + 1)) of $LIST"; exit 1; }
echo "host $(hostname), task $SLURM_ARRAY_TASK_ID: $B"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python code/annotate_hf.py --root annotations --model $T1_RUN/distill_merged --benchmarks "$B" --max-len 32767
echo "=== TASK $SLURM_ARRAY_TASK_ID ($B) DONE ==="
