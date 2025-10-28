#!/bin/bash
#SBATCH -J chehab_eval_running_reduced
#SBATCH -o chehab_eval_running_reduced.%j.out
#SBATCH -e chehab_eval_running_reduced.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-01:30:00

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

export PROJ_ROOT=/scratch/mxa5343/chehab-constr/chehab
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:$PYTHONPATH"

wait_for_file() {
  local path="$1"
  local max_wait="$2"
  local slept=0
  while [ $slept -lt $max_wait ]; do
    [ -f "$path" ] && return 0
    sleep 30
    slept=$((slept+30))
  done
  return 1
}

DATASET_PY='
from RL.fhe_rl.__main__ import load_embeddings_from_config, DATA_DIR
from RL.fhe_rl.test import test_agent
emb,_=load_embeddings_from_config("dynamic")
test_agent(str(DATA_DIR/"reduced_benchmarks.txt"), emb, MODEL_ZIP)
'

run_test () {
  local LABEL="$1"
  local JOBID="$2"
  local BEST="$PROJ_ROOT/eval/best_model_model_${JOBID}/best_model.zip"
  local OUT_XLS="test_results_${LABEL}_reduced.xlsx"

  # prefer best_model.zip, else latest checkpoint; no waiting
  local ZIP="$BEST"
  [ ! -f "$ZIP" ] && ZIP=$(ls -1t "$PROJ_ROOT"/model_${JOBID}_iter_*_real.zip 2>/dev/null | head -1)

  if [ -z "$ZIP" ] || [ ! -f "$ZIP" ]; then
    echo "[SKIP] $LABEL: no best or checkpoint found yet"
    return
  fi

  echo "[RUN] $LABEL -> $ZIP"
  SLURM_JOB_ID="$LABEL" python -u - <<PY
from RL.fhe_rl.__main__ import load_embeddings_from_config, DATA_DIR
from RL.fhe_rl.test import test_agent
emb,_=load_embeddings_from_config("dynamic")
test_agent(str(DATA_DIR/"reduced_benchmarks.txt"), emb, r"""$ZIP""")
PY

  if [ -f test_results.xlsx ]; then
    mv -f test_results.xlsx "$OUT_XLS"
    echo "[OK]  $LABEL Excel: $PROJ_ROOT/$OUT_XLS"
  else
    echo "[ERR] $LABEL produced no test_results.xlsx"
  fi
}

# Running jobs to evaluate (unique labels)
run_test "lag_successk_den2"      "12542361"
run_test "lag_terminal_curr_den2" "12542330"
run_test "lag_terminal_den4_a"    "12542323"
run_test "lag_terminal_den4_b"    "12542118"

echo
echo "[RESULTS]"
ls -1 "$PROJ_ROOT"/test_results_*_reduced.xlsx 2>/dev/null || echo "No Excel files produced."