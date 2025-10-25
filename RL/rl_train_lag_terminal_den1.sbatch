#!/bin/bash
#SBATCH -J chehab_lag_terminal_den4
#SBATCH -o chehab_lag_terminal_den4.%j.out
#SBATCH -e chehab_lag_terminal_den4.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 7-00:00:00

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

export PROJ_ROOT=/scratch/mxa5343/chehab-constr/chehab
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:$PYTHONPATH"

# Training config
export TOTAL_TIMESTEPS=500000
export LAGRANGE_DENOM_FACTOR=1
export EXP_TAG=lag_terminal_den4_reduced

echo "[JOB] tag=$EXP_TAG tt=$TOTAL_TIMESTEPS denom=$LAGRANGE_DENOM_FACTOR dataset=RL/fhe_rl/datasets/reduced_benchmarks.txt"

stdbuf -oL -eL python -u - <<'PY'
from RL.fhe_rl.__main__ import load_embeddings_from_config, DATA_DIR
from RL.fhe_rl.train import train_lagrangian_ppo_agent

emb, _ = load_embeddings_from_config("dynamic")
expr_file = str(DATA_DIR / "reduced_benchmarks.txt")

# TOTAL_TIMESTEPS and LAGRANGE_DENOM_FACTOR are read inside train_lagrangian_ppo_agent
train_lagrangian_ppo_agent(expr_file, emb)
PY