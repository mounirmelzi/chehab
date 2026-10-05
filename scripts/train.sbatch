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

#SBATCH -J train
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.train.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.train.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# Create logs directory if it doesn't exist
mkdir -p "$PROJ_ROOT/logs"

# Run training - uses LAGRANGIAN_PPO with multi-budget support
# Budget options: [240, 300, 1_000_000] - randomly sampled each episode
# Agent receives one-hot encoded budget in observation
stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train 2>&1 | tee -a "$PROJ_ROOT/logs/train.log"
