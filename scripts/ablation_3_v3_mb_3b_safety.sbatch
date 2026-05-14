#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J abl3-v3mb-safe
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.abl3_v3mb_safety.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.abl3_v3mb_safety.%j.txt

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

# Ablation 3: V3 Margin Barrier 3b + Safety
# Purpose: prove V4 budgets > V3 budgets (even with safety)
MODEL="$PROJ_ROOT/RL/eval/best_model_model_14270907_margin_barrier/best_model.zip"
OUTPUT="$PROJ_ROOT/RL/test_results/test_v2_v3_mb_3b_safety.xlsx"
TRAIN_BUDGETS="200,300,9000000"
TEST_BUDGETS="230,236,369,372,9000000"

echo "════════════════════════════════════════════════════════════"
echo "  ABLATION 3: V3 Margin Barrier (3 budgets) + Safety"
echo "  Model:   $MODEL"
echo "  Train:   $TRAIN_BUDGETS"
echo "  Test:    $TEST_BUDGETS"
echo "════════════════════════════════════════════════════════════"

stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
    --model "$MODEL" \
    --budgets "$TEST_BUDGETS" \
    --train_budgets "$TRAIN_BUDGETS" \
    --method margin_barrier \
    --test_mode v2 \
    --output "$OUTPUT"

echo "DONE — $OUTPUT"
