#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-v5
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_v5.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_v5.%j.txt

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
# Test V5 models — NATO-SC with trajectory checkpointing (v2)
# Each model tested on the budgets it was trained on
# Uses --test_mode v2 for safety rollback + feasible/infeasible split
# ═══════════════════════════════════════════════════════════════

run_test_v2() {
    local JOB_ID="$1"
    local NAME="$2"
    local METHOD="$3"
    local TRAIN_BUDGETS="$4"
    local MODEL_DIR="$5"
    local MODEL_PATH="$PROJ_ROOT/RL/eval/${MODEL_DIR}/best_model.zip"
    local OUTPUT="$PROJ_ROOT/RL/test_results/test_v2_${NAME}.xlsx"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Testing (v2): $NAME (job $JOB_ID)"
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
        --test_mode v2 \
        --output "$OUTPUT"

    echo "  DONE — results at $OUTPUT"
}

# ─── V5-A: PPO + NATO-SC [230,233,236,369,9M] ──────────────────
run_test_v2 "14529413" "v5_ppo_natosc_A" \
    nato_sc "230,233,236,369,9000000" \
    "best_model_model_14529413_nato_sc"

# ─── V5-B: PPO + NATO-SC [230,236,369,372,9M] ──────────────────
run_test_v2 "14529414" "v5_ppo_natosc_B" \
    nato_sc "230,236,369,372,9000000" \
    "best_model_model_14529414_nato_sc"

# ─── V5-C: FOCOPS + NATO-SC [230,236,369,9M] ───────────────────
run_test_v2 "14544615" "v5_focops_natosc" \
    nato_sc "230,236,369,9000000" \
    "best_model_model_14544615_nato_sc"

# ─── V5-D: Lagrangian-PID + NATO-SC [230,236,369,9M] ───────────
run_test_v2 "14529416" "v5_lagpid_natosc" \
    nato_sc "230,236,369,9000000" \
    "best_model_model_14529416_nato_sc"


echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL V5 TESTS COMPLETE"
echo "  Results in: $PROJ_ROOT/RL/test_results/"
echo "════════════════════════════════════════════════════════════"
ls -la "$PROJ_ROOT/RL/test_results"/test_v2_v5_*
