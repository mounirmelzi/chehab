#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-14007517
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_14007517.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_14007517.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

mkdir -p "$PROJ_ROOT/logs"

# ═══════════════════════════════════════════════════════════════
# Test model from training job 14007517
# Trained with: default Lagrangian PPO, budgets [240, 300, 1000000]
# Testing on:   all 3 budgets x all benchmark expressions
# ═══════════════════════════════════════════════════════════════

MODEL_PATH="$PROJ_ROOT/RL/eval/best_model_model_14007517/best_model.zip"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Cannot find model at $MODEL_PATH"
    exit 1
fi

TRAIN_BUDGETS="240,300,1000000"
TEST_BUDGETS="240,300,1000000"
OUTPUT_FILE="$PROJ_ROOT/RL/test_results_14007517.xlsx"

echo "════════════════════════════════════════════"
echo "  Model:          $MODEL_PATH"
echo "  Train budgets:  $TRAIN_BUDGETS"
echo "  Test budgets:   $TEST_BUDGETS"
echo "  Output:         $OUTPUT_FILE"
echo "════════════════════════════════════════════"

stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
    --model "$MODEL_PATH" \
    --budgets "$TEST_BUDGETS" \
    --train_budgets "$TRAIN_BUDGETS" \
    --method lagrangian_od_ov \
    --output "$OUTPUT_FILE" \
    2>&1 | tee -a "$PROJ_ROOT/logs/test_14007517.log"
