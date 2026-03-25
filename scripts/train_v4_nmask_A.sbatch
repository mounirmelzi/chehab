#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 7-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J v4-nmask-A
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.v4_nmask_A.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.v4_nmask_A.%j.txt

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

METHOD="noise_masking"
BUDGETS="230,233,236,369,9000000"
TIMESTEPS=5000000
NUM_ENVS=8
DENOM_FACTOR=2
BUDGET_ENCODING="embed"
ENT_COEF=0.01

echo "════════════════════════════════════════════"
echo "  V4 Noise Masking — Group A [230,233,236,369,9M]"
echo "  Method:      $METHOD"
echo "  Budgets:     $BUDGETS"
echo "  Encoding:    $BUDGET_ENCODING"
echo "  Ent coef:    $ENT_COEF"
echo "  Timesteps:   $TIMESTEPS"
echo "  Envs:        $NUM_ENVS"
echo "════════════════════════════════════════════"

stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train \
    --method "$METHOD" \
    --budgets "$BUDGETS" \
    --budget_encoding "$BUDGET_ENCODING" \
    --ent_coef $ENT_COEF \
    --timesteps $TIMESTEPS \
    --num_envs $NUM_ENVS \
    --denom_factor $DENOM_FACTOR \
    2>&1 | tee -a "$PROJ_ROOT/logs/v4_nmask_A.log"
