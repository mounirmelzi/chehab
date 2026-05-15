#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-thresh
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_threshold.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_threshold.%j.txt

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
#  Step 1: Generate threshold benchmark (if not already done)
# ═══════════════════════════════════════════════════════════════
BENCH="./fhe_rl/datasets/threshold_benchmarks.txt"
if [ ! -f "$BENCH" ]; then
    echo "Generating threshold benchmark dataset ..."
    stdbuf -oL -eL python -u ../scripts/generate_threshold_benchmarks.py --output "$BENCH"
    echo ""
fi

echo "Threshold benchmarks: $(wc -l < "$BENCH") expressions"
echo ""

# ═══════════════════════════════════════════════════════════════
#  Step 2: Test all agents on threshold benchmarks
# ═══════════════════════════════════════════════════════════════

TEST_BUDGETS="230,236,369"

run_constrained() {
    local NAME="$1"
    local METHOD="$2"
    local TRAIN_BUDGETS="$3"
    local MODEL_DIR="$4"
    local MODEL_PATH="$PROJ_ROOT/RL/eval/${MODEL_DIR}/best_model.zip"
    local OUTPUT="$PROJ_ROOT/RL/test_results/threshold_${NAME}.xlsx"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Testing: $NAME"
    echo "  Method:  $METHOD | Train: $TRAIN_BUDGETS | Test: $TEST_BUDGETS"
    echo "  Model:   $MODEL_PATH"
    echo "════════════════════════════════════════════════════════════"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIPPED — model not found"
        return
    fi

    stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
        --model "$MODEL_PATH" \
        --budgets "$TEST_BUDGETS" \
        --train_budgets "$TRAIN_BUDGETS" \
        --method "$METHOD" \
        --test_mode v2 \
        --benchmark "$BENCH" \
        --output "$OUTPUT"

    echo "  DONE — $OUTPUT"
}

# ─── Unconstrained baseline (special: runs once, evaluates all budgets) ───
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Testing: Unconstrained (special standalone script)"
echo "════════════════════════════════════════════════════════════"
stdbuf -oL -eL python -u ../scripts/test_unconstrained_threshold.py \
    --benchmark "$BENCH" \
    --budgets "$TEST_BUDGETS" \
    --output "$PROJ_ROOT/RL/test_results/threshold_unconstrained.xlsx"

# ─── V4 agents ────────────────────────────────────────────────
run_constrained "v4_mb_B" \
    margin_barrier "230,236,369,372,9000000" \
    "best_model_model_14385920_margin_barrier"

run_constrained "v4_mb_film_B" \
    margin_barrier "230,236,369,372,9000000" \
    "best_model_model_14385922_margin_barrier"

run_constrained "v4_nmask_B" \
    noise_masking "230,236,369,372,9000000" \
    "best_model_model_14385924_noise_masking"

# ─── V5 agents ────────────────────────────────────────────────
run_constrained "v5_ppo_natosc_A" \
    nato_sc "230,233,236,369,9000000" \
    "best_model_model_14529413_nato_sc"

run_constrained "v5_lagpid_natosc" \
    nato_sc "230,236,369,9000000" \
    "best_model_model_14529416_nato_sc"

run_constrained "v5_focops_natosc" \
    nato_sc "230,236,369,9000000" \
    "best_model_model_14544615_nato_sc"


echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL THRESHOLD TESTS COMPLETE"
echo "  Results in: $PROJ_ROOT/RL/test_results/threshold_*"
echo "════════════════════════════════════════════════════════════"
ls -la "$PROJ_ROOT/RL/test_results"/threshold_*
