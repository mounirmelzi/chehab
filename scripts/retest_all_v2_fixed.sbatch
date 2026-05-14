#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J retest-v2-fix
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.retest_v2_fix.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.retest_v2_fix.%j.txt

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

# ═══════════════════════════════════════════════════════════════
#  RETEST ALL V2 (SAFETY) TESTS — with terminal state bug fix
#  Bug: DummyVecEnv auto-resets on done, old code read from
#       fhe_env (reset state) instead of infos dict (terminal state)
#  Fix: env.py adds cost to info, test.py reads from infos[0]
# ═══════════════════════════════════════════════════════════════

run_test_v2() {
    local NAME="$1"
    local METHOD="$2"
    local BUDGETS="$3"
    local MODEL_PATH="$4"
    local OUTPUT="$PROJ_ROOT/RL/test_results/test_v2_${NAME}.xlsx"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Testing (v2 FIXED): $NAME"
    echo "  Method:  $METHOD"
    echo "  Budgets: $BUDGETS"
    echo "  Model:   $MODEL_PATH"
    echo "  Output:  $OUTPUT"
    echo "════════════════════════════════════════════════════════════"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIPPED — model not found: $MODEL_PATH"
        return
    fi

    stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
        --model "$MODEL_PATH" \
        --budgets "$BUDGETS" \
        --train_budgets "$BUDGETS" \
        --method "$METHOD" \
        --test_mode v2 \
        --output "$OUTPUT"

    echo "  DONE — $OUTPUT"
}

# ─── 1. V4 MB_B + Safety (our best agent) ────────────────────────
run_test_v2 "v4_mb_B_safety" \
    margin_barrier \
    "230,236,369,372,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14385920_margin_barrier/best_model.zip"

# ─── 2. V5-A: PPO + NATO-SC ─────────────────────────────────────
run_test_v2 "v5_ppo_natosc_A" \
    nato_sc \
    "230,233,236,369,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14529413_nato_sc/best_model.zip"

# ─── 3. V5-B: PPO + NATO-SC ─────────────────────────────────────
run_test_v2 "v5_ppo_natosc_B" \
    nato_sc \
    "230,236,369,372,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14529414_nato_sc/best_model.zip"

# ─── 4. V5-C: FOCOPS + NATO-SC ──────────────────────────────────
run_test_v2 "v5_focops_natosc" \
    nato_sc \
    "230,236,369,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14544615_nato_sc/best_model.zip"

# ─── 5. V5-D: Lagrangian-PID + NATO-SC ──────────────────────────
run_test_v2 "v5_lagpid_natosc" \
    nato_sc \
    "230,236,369,9000000" \
    "$PROJ_ROOT/RL/eval/best_model_model_14529416_nato_sc/best_model.zip"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL V2 RETESTS COMPLETE"
echo "════════════════════════════════════════════════════════════"
ls -la "$PROJ_ROOT/RL/test_results"/test_v2_*
