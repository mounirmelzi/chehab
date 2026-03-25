#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-aug-all
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_aug_all.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_aug_all.%j.txt

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
# Pre-flight: verify augmented benchmarks exist
# ═══════════════════════════════════════════════════════════════
BENCH="./fhe_rl/datasets/benchmarks.txt"
N_LINES=$(wc -l < "$BENCH")
echo "Benchmark file: $BENCH ($N_LINES lines)"

if [ "$N_LINES" -lt 60 ]; then
    echo "ERROR: benchmarks.txt has only $N_LINES lines (expected ~70 after augmentation)."
    echo "Did you run augment_benchmarks.sbatch first?"
    exit 1
fi

echo "Benchmark file OK — proceeding with testing"
echo ""

# ═══════════════════════════════════════════════════════════════
# Test function — same pattern as other test scripts
# ═══════════════════════════════════════════════════════════════

run_test() {
    local NAME="$1"
    local METHOD="$2"
    local TRAIN_BUDGETS="$3"
    local MODEL_PATH="$4"
    local OUTPUT="$PROJ_ROOT/RL/test_results/test_aug_${NAME}.xlsx"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Testing: $NAME"
    echo "  Method:  $METHOD"
    echo "  Budgets: $TRAIN_BUDGETS"
    echo "  Model:   $MODEL_PATH"
    echo "  Output:  $OUTPUT"
    echo "════════════════════════════════════════════════════════════"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIPPED — model not found: $MODEL_PATH"
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

# ═══════════════════════════════════════════════════════════════
# 1. UNCONSTRAINED PPO BASELINE
#    Model: original Bilel/Raouf agent (no constraints)
#    Train budgets: default [240, 300, 1000000]
#    Method: none
# ═══════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SECTION 1: UNCONSTRAINED PPO BASELINE                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"

run_test "unconstrained_ppo" \
    none "240,300,1000000" \
    "$PROJ_ROOT/RL/fhe_rl/trained_models/agent_dynamic_llm_data.zip"


# ═══════════════════════════════════════════════════════════════
# 2. V3 AGENTS (best performers from V3 generation)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SECTION 2: V3 AGENTS                                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# 6-budget models
run_test "v3_mb_6b" \
    margin_barrier "180,200,220,240,300,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270905_margin_barrier/best_model.zip"

run_test "v3_nmask_6b" \
    noise_masking "180,200,220,240,300,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270906_noise_masking/best_model.zip"

# 3-budget models
run_test "v3_mb_3b" \
    margin_barrier "200,300,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270907_margin_barrier/best_model.zip"

run_test "v3_nmask_3b" \
    noise_masking "200,300,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270909_noise_masking/best_model.zip"

run_test "v3_nmask_cur_3b" \
    noise_masking "200,300,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270912_noise_masking/best_model.zip"

run_test "v3_mb_film_3b" \
    margin_barrier "200,300,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270914_margin_barrier/best_model.zip"

# 5-budget models
run_test "v3_mb_5b" \
    margin_barrier "180,200,220,240,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270908_margin_barrier/best_model.zip"

run_test "v3_nmask_5b" \
    noise_masking "180,200,220,240,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270911_noise_masking/best_model.zip"

run_test "v3_mb_film_5b" \
    margin_barrier "180,200,220,240,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14270915_margin_barrier/best_model.zip"


# ═══════════════════════════════════════════════════════════════
# 3. V4 AGENTS (trained on augmented high-noise dataset)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SECTION 3: V4 AGENTS                                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# Group A: Margin Barrier [230,233,236,369,9M]
run_test "v4_mb_A" \
    margin_barrier "230,233,236,369,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385919_margin_barrier/best_model.zip"

# Group B: Margin Barrier [230,236,369,372,9M]
run_test "v4_mb_B" \
    margin_barrier "230,236,369,372,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385920_margin_barrier/best_model.zip"

# Group A: MB+FiLM [230,233,236,369,9M]
run_test "v4_mb_film_A" \
    margin_barrier "230,233,236,369,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385921_margin_barrier/best_model.zip"

# Group B: MB+FiLM [230,236,369,372,9M]
run_test "v4_mb_film_B" \
    margin_barrier "230,236,369,372,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385922_margin_barrier/best_model.zip"

# Group A: Noise Masking [230,233,236,369,9M]
run_test "v4_nmask_A" \
    noise_masking "230,233,236,369,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385923_noise_masking/best_model.zip"

# Group B: Noise Masking [230,236,369,372,9M]
run_test "v4_nmask_B" \
    noise_masking "230,236,369,372,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385924_noise_masking/best_model.zip"

# Group C: MB+FiLM [172,230,236,369,9M]
run_test "v4_mb_film_C" \
    margin_barrier "172,230,236,369,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385925_margin_barrier/best_model.zip"

# Group C: Noise Masking [172,233,369,372,9M]
run_test "v4_nmask_C" \
    noise_masking "172,233,369,372,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385926_noise_masking/best_model.zip"


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL TESTS COMPLETE ON AUGMENTED BENCHMARKS"
echo "  Results in: $PROJ_ROOT/RL/test_results/"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Augmented test results (test_aug_*):"
ls -la "$PROJ_ROOT/RL/test_results/test_aug_"* 2>/dev/null || echo "  No test_aug_* files found"
echo ""
echo "Summary of tested agents:"
echo "  1. Unconstrained PPO baseline"
echo "  2. V3 agents: mb_6b, nmask_6b, mb_3b, nmask_3b, nmask_cur_3b, mb_film_3b, mb_5b, nmask_5b, mb_film_5b"
echo "  3. V4 agents: mb_A, mb_B, mb_film_A, mb_film_B, nmask_A, nmask_B, mb_film_C, nmask_C"
