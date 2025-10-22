#!/bin/bash
#SBATCH -J chehab_lag_excess_den4
#SBATCH -o chehab_lag_excess_den4.%j.out
#SBATCH -e chehab_lag_excess_den4.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --reservation=c2
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

# Experiment: Excess-only penalty with budget=200 and denom=4
# Rationale:
# - Penalize only noise beyond (budget - 10%) instead of every bit, to prevent
#   early collapse of exploration while still enforcing the constraint.
# - Budget=200 gives the agent more headroom; the wrapper normalizes penalties
#   by budget so scales remain comparable; λ keeps avg noise near the target.
# - Expect longer episodes than baseline constrained runs, avg_noise converging
#   just below ~200, and higher eval/mean_reward at comparable constraint level.

export NOISE_BUDGET=200
export LAGRANGE_DENOM_FACTOR=4
stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run_lag_excess_den4.log


