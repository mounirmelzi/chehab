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

#SBATCH -J train_budget_one_hot
#SBATCH -o /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/output.train_budget_one_hot.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/error.train_budget_one_hot.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-picked-agent/chehab"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# Budget Strategy 1: Fixed 10-budget one-hot feature
export BUDGET_STRATEGY=one_hot

# Training configuration
export TOTAL_TIMESTEPS=5000000  # 5 million timesteps
export DENOM_FACTOR=2  # Lower = more frequent lambda updates (default: 4)

mkdir -p "$PROJ_ROOT/logs"
stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train \
  2>&1 | tee -a "$PROJ_ROOT/logs/train_budget_one_hot.log"

