#!/bin/bash
#SBATCH -J chehab_lag_perstep_curr_den1
#SBATCH -o chehab_lag_perstep_curr_den1.%j.out
#SBATCH -e chehab_lag_perstep_curr_den1.%j.err
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

# Experiment: Lagrangian PPO with Per-Step Penalty + Budget Curriculum (denom=1)
# Method: Per-Step Penalty with Constraint Curriculum
# - Per-step penalty: If current expression noise > current budget, apply penalty every step
# - Budget curriculum: Budget tightens from BUDGET_START to BUDGET_FINAL across training
# - Penalty = λ × (excess / budget) where excess = current_noise - current_budget
# - Terminal penalty: Additional penalty at episode end if final noise > budget
# - Expected: Early exploration with high budget, gradual constraint enforcement
# - LAGRANGE_DENOM_FACTOR=1 means λ updates every ~31k steps

export LAGRANGE_DENOM_FACTOR=1
export BUDGET_START=5000
export BUDGET_FINAL=240
stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run_lag_perstep_curr_den1.log