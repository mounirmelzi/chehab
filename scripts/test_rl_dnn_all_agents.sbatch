#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J rl-dnn-all
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.rl_dnn_all_agents.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.rl_dnn_all_agents.%j.txt

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
#  Comprehensive RL Ablation on DNN Benchmarks
#
#  Tests ALL agents (6 constrained + 1 unconstrained) on the
#  combined DNN benchmark set (original + scaled = 11 expressions)
#  using --test_mode v2 (safety rollback + feasible/infeasible split)
# ═══════════════════════════════════════════════════════════════

# Build combined DNN benchmark file
DNN_ORIG="./fhe_rl/datasets/dnn_benchmarks.txt"
DNN_SCALED="./fhe_rl/datasets/dnn_benchmarks_scaled.txt"
COMBINED="./fhe_rl/datasets/dnn_benchmarks_combined.txt"

cat "$DNN_ORIG" > "$COMBINED"
if [ -f "$DNN_SCALED" ]; then
    cat "$DNN_SCALED" >> "$COMBINED"
fi

BENCH="$COMBINED"
TEST_BUDGETS="230,236,369,9000000"

TOTAL_EXPR=$(wc -l < "$BENCH")
echo "═══════════════════════════════════════════════════════════"
echo "  DNN Ablation: $TOTAL_EXPR expressions"
echo "  Test budgets: $TEST_BUDGETS"
echo "  Benchmark file: $BENCH"
echo "═══════════════════════════════════════════════════════════"
echo ""

PASS=0
FAIL=0

run_constrained() {
    local NAME="$1"
    local METHOD="$2"
    local TRAIN_BUDGETS="$3"
    local MODEL_DIR="$4"
    local MODEL_PATH="$PROJ_ROOT/RL/eval/${MODEL_DIR}/best_model.zip"
    local OUTPUT="$PROJ_ROOT/RL/test_results/dnn_combined_${NAME}.xlsx"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Testing: $NAME"
    echo "  Method:  $METHOD | Train budgets: $TRAIN_BUDGETS"
    echo "  Model:   $MODEL_PATH"
    echo "════════════════════════════════════════════════════════════"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIPPED — model not found at $MODEL_PATH"
        FAIL=$((FAIL + 1))
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

    if [ $? -eq 0 ]; then
        echo "  DONE — $OUTPUT"
        PASS=$((PASS + 1))
    else
        echo "  FAILED"
        FAIL=$((FAIL + 1))
    fi
}

# ─── 1. Unconstrained PPO baseline ──────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  1/7: Unconstrained PPO (legacy baseline)"
echo "════════════════════════════════════════════════════════════"
stdbuf -oL -eL python -u ../scripts/test_unconstrained_threshold.py \
    --benchmark "$BENCH" \
    --budgets "$TEST_BUDGETS" \
    --output "$PROJ_ROOT/RL/test_results/dnn_combined_unconstrained.xlsx"
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi

# ─── 2. V4 Margin Barrier B ─────────────────────────────────
run_constrained "v4_mb_B" \
    margin_barrier "230,236,369,372,9000000" \
    "best_model_model_14385920_margin_barrier"

# ─── 3. V4 Noise Masking B ──────────────────────────────────
run_constrained "v4_nmask_B" \
    noise_masking "230,236,369,372,9000000" \
    "best_model_model_14385924_noise_masking"

# ─── 4. V4 MB + FiLM B ─────────────────────────────────────
run_constrained "v4_mb_film_B" \
    margin_barrier "230,236,369,372,9000000" \
    "best_model_model_14385922_margin_barrier"

# ─── 5. V5 FOCOPS + NATO-SC ─────────────────────────────────
run_constrained "v5_focops_natosc" \
    nato_sc "230,236,369,9000000" \
    "best_model_model_14544615_nato_sc"

# ─── 6. V5 PPO + NATO-SC A ──────────────────────────────────
run_constrained "v5_ppo_natosc_A" \
    nato_sc "230,233,236,369,9000000" \
    "best_model_model_14529413_nato_sc"

# ─── 7. V5 Lag-PID + NATO-SC (BEST) ─────────────────────────
run_constrained "v5_lagpid_natosc" \
    nato_sc "230,236,369,9000000" \
    "best_model_model_14529416_nato_sc"


echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL DNN ABLATION TESTS COMPLETE"
echo "  Passed: $PASS / $((PASS + FAIL))"
echo "  Failed: $FAIL / $((PASS + FAIL))"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Result files:"
ls -la "$PROJ_ROOT/RL/test_results"/dnn_combined_* 2>/dev/null
