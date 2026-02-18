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

#SBATCH -J training-10budgets-8envs-denom2
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.train_10budgets.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.train_10budgets.%j.txt

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

# 10 Budgets Training: [240, 300, 400, 500, 600, 700, 800, 900, 1000, 9000000]
# Total timesteps: 5,000,000
# Environments: 8
# Denom factor: 2 (more frequent lambda updates)
BUDGETS="240,300,400,500,600,700,800,900,1000,9000000"
TIMESTEPS=5000000
NUM_ENVS=8
DENOM_FACTOR=2

echo "Starting training with 10 budgets: $BUDGETS"
echo "Timesteps: $TIMESTEPS, Environments: $NUM_ENVS, Denom factor: $DENOM_FACTOR"

stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train \
    --budgets "$BUDGETS" \
    --timesteps $TIMESTEPS \
    --num_envs $NUM_ENVS \
    --denom_factor $DENOM_FACTOR \
    2>&1 | tee -a "$PROJ_ROOT/logs/train_10budgets.log"
