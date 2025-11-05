#!/bin/bash
#SBATCH -J chehab_lag_successk_den2_k3_llm_data
#SBATCH -o chehab_lag_successk_den2_k3_llm_data.%j.out
#SBATCH -e chehab_lag_successk_den2_k3_llm_data.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 7-00:00:00

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr/chehab"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# Experiment: Lagrangian PPO with success bonus + k-consecutive penalty (denom=2)
export LAGRANGE_DENOM_FACTOR=2
export CONSTRAINT_WRAPPER=success_k
export K_CONSECUTIVE=3
export BUDGET_START=5000
export BUDGET_FINAL=240

stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run_lag_successk_den2_k3_llm_data.log