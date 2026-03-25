#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-v3
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_v3.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_v3.%j.txt

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
mkdir -p "$PROJ_ROOT/RL/test_results"

# ═══════════════════════════════════════════════════════════════
# Test V3 models — moderate budget ranges
# ═══════════════════════════════════════════════════════════════

run_test() {
    local JOB_ID="$1"
    local NAME="$2"
    local METHOD="$3"
    local TRAIN_BUDGETS="$4"
    local MODEL_DIR="$5"
    local MODEL_PATH="$PROJ_ROOT/RL/eval/${MODEL_DIR}/best_model.zip"
    local OUTPUT="$PROJ_ROOT/RL/test_results/test_${NAME}.xlsx"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Testing: $NAME (job $JOB_ID)"
    echo "  Method:  $METHOD"
    echo "  Budgets: $TRAIN_BUDGETS"
    echo "  Model:   $MODEL_PATH"
    echo "  Output:  $OUTPUT"
    echo "════════════════════════════════════════════════════════════"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIPPED — model not found (training may not be done yet)"
        return
    fi

    stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
        --model "$MODEL_PATH" \
        --budgets "$TRAIN_BUDGETS" \
        --train_budgets "$TRAIN_BUDGETS" \
        --method "$METHOD" \
        --output "$OUTPUT"

    echo "  DONE — results at $OUTPUT"
}

# ─── 6-budget models (top 2 methods) ──────────────────────────
run_test 14270905 "v3_mb_6b"         margin_barrier "180,200,220,240,300,9000000" \
    "best_model_model_14270905_margin_barrier"

run_test 14270906 "v3_nmask_6b"      noise_masking  "180,200,220,240,300,9000000" \
    "best_model_model_14270906_noise_masking"

# ─── 3-budget models ──────────────────────────────────────────
run_test 14270907 "v3_mb_3b"         margin_barrier "200,300,9000000" \
    "best_model_model_14270907_margin_barrier"

run_test 14270909 "v3_nmask_3b"      noise_masking  "200,300,9000000" \
    "best_model_model_14270909_noise_masking"

run_test 14270912 "v3_nmask_cur_3b"  noise_masking  "200,300,9000000" \
    "best_model_model_14270912_noise_masking"

run_test 14270914 "v3_mb_film_3b"    margin_barrier "200,300,9000000" \
    "best_model_model_14270914_margin_barrier"

# ─── 5-budget models ──────────────────────────────────────────
run_test 14270908 "v3_mb_5b"         margin_barrier "180,200,220,240,9000000" \
    "best_model_model_14270908_margin_barrier"

run_test 14270911 "v3_nmask_5b"      noise_masking  "180,200,220,240,9000000" \
    "best_model_model_14270911_noise_masking"

run_test 14270915 "v3_mb_film_5b"    margin_barrier "180,200,220,240,9000000" \
    "best_model_model_14270915_margin_barrier"


echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL V3 TESTS COMPLETE"
echo "  Results in: $PROJ_ROOT/RL/test_results/"
echo "════════════════════════════════════════════════════════════"
ls -la "$PROJ_ROOT/RL/test_results/"
