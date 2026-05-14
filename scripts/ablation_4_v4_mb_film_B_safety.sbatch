#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J abl4-film-safe
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.abl4_film_safety.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.abl4_film_safety.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

mkdir -p "$PROJ_ROOT/RL/test_results"

# Ablation 4: V4 MB FiLM B + Safety
# Purpose: prove raw encoding + safety >= FiLM encoding + safety
MODEL="$PROJ_ROOT/RL/eval/best_model_model_14385922_margin_barrier/best_model.zip"
OUTPUT="$PROJ_ROOT/RL/test_results/test_v2_v4_mb_film_B_safety.xlsx"
BUDGETS="230,236,369,372,9000000"

echo "════════════════════════════════════════════════════════════"
echo "  ABLATION 4: V4 MB FiLM B + Safety Rollback"
echo "  Model:   $MODEL"
echo "  Budgets: $BUDGETS"
echo "════════════════════════════════════════════════════════════"

stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
    --model "$MODEL" \
    --budgets "$BUDGETS" \
    --train_budgets "$BUDGETS" \
    --method margin_barrier \
    --test_mode v2 \
    --output "$OUTPUT"

echo "DONE — $OUTPUT"
