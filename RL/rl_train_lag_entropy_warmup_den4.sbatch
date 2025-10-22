#!/bin/bash
#SBATCH -J chehab_lag_entropy_warmup_den4
#SBATCH -o chehab_lag_entropy_warmup_den4.%j.out
#SBATCH -e chehab_lag_entropy_warmup_den4.%j.err
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

# Experiment: Lagrangian PPO with exploration boost + lambda warmup (denom=4)
# What this job does (code is already implemented in RL/fhe_rl/train.py & wrappers.py):
# 1) Exploration boost (method 2): PPO entropy coefficient starts high (0.2)
#    and decays to 0.05 over ~70% of training via EntCoefScheduler. This
#    emulates epsilon-greedy to encourage early exploration without hacking PPO.
# 2) Lambda warmup + gentler controller: base lambda starts at 0.02 and ramps
#    to full over the first 3 outer iterations; controller gain reduced to 0.005
#    to reduce overshoot. We log both base and effective lambda each iteration.
# 3) Noise budget: set to 200 in code; train.py uses noise_threshold=200 for
#    lambda updates; per-step penalty remains reward -= lambda * noise_used.
#    The budget value is for validation/dual updates and for future reference.
# Expected outcome: longer episodes early (no collapse), smoother approach of
#    Lagrange/avg_noise toward ~200, and higher eval/mean_reward under the
#    constraint compared to the previous constrained runs.

export LAGRANGE_DENOM_FACTOR=4
stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run_lag_entropy_warmup_den4.log


