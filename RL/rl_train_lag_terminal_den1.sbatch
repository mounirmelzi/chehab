#!/bin/bash
#SBATCH -J chehab_lag_terminal_den1
#SBATCH -o chehab_lag_terminal_den1.%j.out
#SBATCH -e chehab_lag_terminal_den1.%j.err
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

# Experiment: Lagrangian PPO with Terminal-Only Penalty (denom=1)
# Method: Terminal-Only Penalty
# - Only penalizes at episode end if final expression noise > budget (200)
# - No per-step penalties during exploration - agent can explore freely
# - Uses final expression noise (not cumulative) for penalty calculation
# - Penalty = λ × max(0, final_noise - budget) / budget
# - Expected: Longer episodes, better exploration, constraint compliance at end
# - LAGRANGE_DENOM_FACTOR=1 means more frequent λ updates (every ~31k steps)

export LAGRANGE_DENOM_FACTOR=1
stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run_lag_terminal_den1.log