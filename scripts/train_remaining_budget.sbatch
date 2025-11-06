#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 7-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J train_remaining_budget_300
#SBATCH -o /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/output.train_remaining_budget_300.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/error.train_remaining_budget_300.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-picked-agent/chehab"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# Environment variables for training configuration
export TOTAL_TIMESTEPS=2000000
export LAGRANGE_DENOM_FACTOR=4
export NOISE_BUDGET=300
export TRAIN_DATASET=benchmarks.txt
# Use default wrapper (ON_DONE + ON_VIOLATION) - no LAGRANGIAN_WRAPPER set

# Create logs directory if it doesn't exist (before SLURM tries to write)
mkdir -p "$PROJ_ROOT/logs"

# Run training from RL directory
# This trains Lagrangian PPO with remaining budget as observation feature
# Noise threshold: 300, Wrapper: default (ON_DONE + ON_VIOLATION)
stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train 2>&1 | tee -a "$PROJ_ROOT/logs/train_remaining_budget_300.log"

