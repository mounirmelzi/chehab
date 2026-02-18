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

#SBATCH -J lagrangian_perstep-5b-12envs
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.train_lagrangian_perstep_5b.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.train_lagrangian_perstep_5b.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

mkdir -p "$PROJ_ROOT/logs"

# ═══════════════════════════════════════════════════════════════
# Method:     lagrangian_perstep
# Budgets:    5 budgets [300, 400, 500, 1K, 9M]
# Timesteps:  3,000,000
# Envs:       12  |  Cores: 16  |  Denom: 1
# ═══════════════════════════════════════════════════════════════
METHOD="lagrangian_perstep"
BUDGETS="300,400,500,1000,9000000"
TIMESTEPS=3000000
NUM_ENVS=12
DENOM_FACTOR=1

echo "════════════════════════════════════════════"
echo "  Method:      $METHOD"
echo "  Budgets:     $BUDGETS"
echo "  Timesteps:   $TIMESTEPS"
echo "  Envs:        $NUM_ENVS"
echo "  Denom:       $DENOM_FACTOR"
echo "════════════════════════════════════════════"

stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train \
    --method "$METHOD" \
    --budgets "$BUDGETS" \
    --timesteps $TIMESTEPS \
    --num_envs $NUM_ENVS \
    --denom_factor $DENOM_FACTOR \
    2>&1 | tee -a "$PROJ_ROOT/logs/train_lagrangian_perstep_5b.log"
