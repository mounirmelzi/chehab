#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-v4
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_v4.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_v4.%j.txt

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
# Test V4 models — trained on augmented high-noise dataset
# Each model tested on the budgets it was trained on
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

# ─── Group A: Margin Barrier [230,233,236,369,9M] ─────────────
run_test 14385919 "v4_mb_A" \
    margin_barrier "230,233,236,369,9000000" \
    "best_model_model_14385919_margin_barrier"

# ─── Group B: Margin Barrier [230,236,369,372,9M] ─────────────
run_test 14385920 "v4_mb_B" \
    margin_barrier "230,236,369,372,9000000" \
    "best_model_model_14385920_margin_barrier"

# ─── Group A: MB+FiLM [230,233,236,369,9M] ────────────────────
run_test 14385921 "v4_mb_film_A" \
    margin_barrier "230,233,236,369,9000000" \
    "best_model_model_14385921_margin_barrier"

# ─── Group B: MB+FiLM [230,236,369,372,9M] ────────────────────
run_test 14385922 "v4_mb_film_B" \
    margin_barrier "230,236,369,372,9000000" \
    "best_model_model_14385922_margin_barrier"

# ─── Group A: Noise Masking [230,233,236,369,9M] ──────────────
run_test 14385923 "v4_nmask_A" \
    noise_masking "230,233,236,369,9000000" \
    "best_model_model_14385923_noise_masking"

# ─── Group B: Noise Masking [230,236,369,372,9M] ──────────────
run_test 14385924 "v4_nmask_B" \
    noise_masking "230,236,369,372,9000000" \
    "best_model_model_14385924_noise_masking"

# ─── Group C: MB+FiLM [172,230,236,369,9M] ────────────────────
run_test 14385925 "v4_mb_film_C" \
    margin_barrier "172,230,236,369,9000000" \
    "best_model_model_14385925_margin_barrier"

# ─── Group C: Noise Masking [172,233,369,372,9M] ──────────────
run_test 14385926 "v4_nmask_C" \
    noise_masking "172,233,369,372,9000000" \
    "best_model_model_14385926_noise_masking"


echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL V4 TESTS COMPLETE"
echo "  Results in: $PROJ_ROOT/RL/test_results/"
echo "════════════════════════════════════════════════════════════"
ls -la "$PROJ_ROOT/RL/test_results/test_v4_*"
