#!/bin/bash
#SBATCH -J chehab_lag_living_bonus_den4
#SBATCH -o chehab_lag_living_bonus_den4.%j.out
#SBATCH -e chehab_lag_living_bonus_den4.%j.err
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

# Experiment: Lagrangian PPO with living-bonus shaping (denom=4)
# Implementation details (already in code):
# - wrappers.py: At each step, we still subtract the Lagrangian penalty
#   reward -= lambda * noise_used. In addition, we add a small living bonus
#   only while the episode's cumulative noise is safely under (budget - margin),
#   with budget=200 and margin=10. The bonus scales with remaining headroom:
#       bonus = living_bonus_base * headroom / (budget - margin)
#   (living_bonus_base=0.5, tuned small so it nudges exploration without
#   overpowering penalties). cum_noise resets on episode end.
# - train.py: noise_threshold=200 for the dual update; lambda warmup and gentler
#   controller (gain=0.005) stabilize the dual; entropy schedule is present by
#   default—if you want a pure living-bonus run, set ent_coef constant in code.
# Expected effect: more steps while safely under the budget, smoother approach of
#   average noise toward ~200, and higher eval reward under the constraint.

export LAGRANGE_DENOM_FACTOR=4
stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run_lag_living_bonus_den4.log


