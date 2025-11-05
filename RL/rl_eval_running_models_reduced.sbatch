#!/bin/bash
#SBATCH -J chehab_eval_reduced
#SBATCH -o chehab_eval_reduced.%j.out
#SBATCH -e chehab_eval_reduced.%j.err
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
emb,_=load_embeddings_from_config("dynamic")
test_agent(str(DATA_DIR/"reduced_benchmarks.txt"), emb, MODEL_ZIP)
'

run_test () {
  local LABEL="$1"
  local MODEL_ZIP="$2"
  local OUT_XLS="test_results_reduced_${LABEL}.xlsx"

  if [ ! -f "$MODEL_ZIP" ]; then
    echo "[SKIP] $LABEL missing: $MODEL_ZIP"
    return
  fi

  echo "[RUN] $LABEL -> $MODEL_ZIP"
  # override SLURM_JOB_ID for sheet naming in Excel
  SLURM_JOB_ID="$LABEL" python -u - <<PY
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

# Test the newly completed models (with best_model.zip)
run_test "lag_terminal_den2"           "$PROJ_ROOT/eval/best_model_model_12683646/best_model.zip"
run_test "lag_terminal_den1"           "$PROJ_ROOT/eval/best_model_model_12683648/best_model.zip"
run_test "lag_terminal_den2_240"        "$PROJ_ROOT/eval/best_model_model_12683654/best_model.zip"
run_test "lag_terminal_den2_300"       "$PROJ_ROOT/eval/best_model_model_12683663/best_model.zip"
run_test "lag_perstep_curr_den1"       "$PROJ_ROOT/eval/best_model_model_12683683/best_model.zip"
run_test "lag_successk_den2"           "$PROJ_ROOT/eval/best_model_model_12683708/best_model.zip"
run_test "lag_excess_den2_240"         "$PROJ_ROOT/eval/best_model_model_12683747/best_model.zip"
run_test "lag_excess_den2_300"         "$PROJ_ROOT/eval/best_model_model_12683761/best_model.zip"

echo
echo "[RESULTS]"
ls -1 "$PROJ_ROOT"/test_results_reduced_*.xlsx 2>/dev/null || echo "No Excel files produced."