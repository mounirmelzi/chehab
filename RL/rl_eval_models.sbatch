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
emb,_=load_embeddings_from_config("dynamic")
test_agent(str(DATA_DIR/"benchmarks.txt"), emb, MODEL_ZIP)
'

run_test () {
  local LABEL="$1"
  local MODEL_ZIP="$2"
  local OUT_XLS="test_results_${LABEL}.xlsx"

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

# Known constrained runs (exact paths)
run_test "lag_excess_den4"        "$PROJ_ROOT/eval/best_model_model_12493999/best_model.zip"
run_test "lag_entropy_warmup_den4" "$PROJ_ROOT/eval/best_model_model_12494155/best_model.zip"
run_test "lag_living_bonus_den4"   "$PROJ_ROOT/eval/best_model_model_12494168/best_model.zip"

# Optional PPO 1M (set path if different)
PPO_1M_ZIP="$PROJ_ROOT/eval/best_model_model_12457801/best_model.zip"
run_test "ppo_1m" "$PPO_1M_ZIP"

echo
echo "[RESULTS]"
ls -1 "$PROJ_ROOT"/test_results_*.xlsx 2>/dev/null || echo "No Excel files produced."