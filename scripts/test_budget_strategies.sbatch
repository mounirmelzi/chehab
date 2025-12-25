#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1-00:00:00

#SBATCH -J test_budget_strategies
#SBATCH -o /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/output.test_budget_strategies.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/error.test_budget_strategies.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-picked-agent/chehab"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

mkdir -p "$PROJ_ROOT/logs"

test_model() {
    local JOB_ID="$1"
    local LABEL="$2"
    local STRATEGY="$3"
    local BUDGET_LIST="$4"   # e.g. "200,300,inf"
    local MODEL_PATH="$PROJ_ROOT/RL/eval/best_model_model_${JOB_ID}/best_model.zip"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "[SKIP] $LABEL: Model not found at $MODEL_PATH"
        return
    fi

    echo "[TEST] $LABEL (Job $JOB_ID, strategy=$STRATEGY, budgets=${BUDGET_LIST:-default})"

    CMD="SLURM_JOB_ID=${LABEL} MODEL_PATH=${MODEL_PATH} BUDGET_STRATEGY=${STRATEGY} stdbuf -oL -eL python -m fhe_rl --tokenizer_type dynamic test --model_path ${MODEL_PATH} --budget_strategy ${STRATEGY}"
    if [ -n "$BUDGET_LIST" ]; then
        CMD+=" --budgets \"${BUDGET_LIST}\""
    fi

    eval $CMD
    local STATUS=$?

    if [ $STATUS -eq 0 ]; then
        if [ -f "test_results.xlsx" ]; then
            mv -f test_results.xlsx "test_results_${LABEL}.xlsx"
            echo "[OK] Results saved to test_results_${LABEL}.xlsx"
        else:
            echo "[WARN] Test completed but test_results.xlsx not found"
        fi
    else
        echo "[ERROR] Test failed with exit code $STATUS"
    fi
}

# Test the 3 agents (update JOB_IDs when training completes)
# Strategy 1: One-hot encoding (10 fixed budgets)
test_model "13218745" "budget_one_hot" "one_hot" "200,300,350,400,500,600,700,1000,9000000"

# Strategy 2: Budget threshold with learned embedding (dynamic budgets)
test_model "13218168" "budget_threshold" "budget_threshold" "200,300,400,500,600,700,800,1000,inf"

# Strategy 3: Remaining budget with learned embedding (dynamic budgets)
test_model "13217905" "budget_remaining" "remaining_budget" "200,300,400,500,600,700,800,1000,inf"

