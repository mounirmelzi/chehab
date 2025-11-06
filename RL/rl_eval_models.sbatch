#!/bin/bash
#SBATCH -J chehab_eval_suite
#SBATCH -o chehab_eval_suite.%j.out
#SBATCH -e chehab_eval_suite.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:00:00

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

export PROJ_ROOT=/scratch/mxa5343/chehab-constr/chehab
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:$PYTHONPATH"

DATASET_PY='
from RL.fhe_rl.__main__ import load_embeddings_from_config, DATA_DIR
from RL.fhe_rl.test import test_agent
import os
emb,_=load_embeddings_from_config("dynamic")
# Get test budget from environment variable (for models trained with budget feature)
test_budget = os.getenv("TEST_BUDGET")
if test_budget:
    test_budget = float(test_budget)
test_agent(str(DATA_DIR/"benchmarks.txt"), emb, MODEL_ZIP, test_budget=test_budget)
'

run_test () {
  local LABEL="$1"
  local MODEL_ZIP="$2"
  local TEST_BUDGET="${3:-}"  # Optional budget parameter
  local OUT_XLS="test_results_${LABEL}.xlsx"

  if [ ! -f "$MODEL_ZIP" ]; then
    echo "[SKIP] $LABEL missing: $MODEL_ZIP"
    return
  fi

  echo "[RUN] $LABEL -> $MODEL_ZIP"
  if [ -n "$TEST_BUDGET" ]; then
    echo "  Using budget: $TEST_BUDGET"
  fi
  # override SLURM_JOB_ID for sheet naming in Excel
  # Pass TEST_BUDGET as environment variable if provided
  SLURM_JOB_ID="$LABEL" TEST_BUDGET="$TEST_BUDGET" python -u - <<PY
MODEL_ZIP="$MODEL_ZIP"
$DATASET_PY
PY

  if [ -f test_results.xlsx ]; then
    mv -f test_results.xlsx "$OUT_XLS"
    echo "[OK]  $LABEL Excel: $PROJ_ROOT/$OUT_XLS"
  else
    echo "[ERR] $LABEL produced no test_results.xlsx"
  fi
}

# Test models with budget feature
# Format: run_test "label" "model_path" "budget" (budget is optional, use 240, 300, or 1000000)
# If budget is not specified, it will be randomly selected for each expression

# Models trained with budget feature (from chehab-constr)
run_test "budget_feature_12761154_budget1000" "$PROJ_ROOT/eval/best_model_model_12761154/best_model.zip" "1000"
run_test "budget_feature_12763203_budget1000000" "$PROJ_ROOT/eval/best_model_model_12763203/best_model.zip" "1000000"


echo
echo "[RESULTS]"
ls -1 "$PROJ_ROOT"/test_results_*.xlsx 2>/dev/null || echo "No Excel files produced."