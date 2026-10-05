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

#SBATCH -J v2-embed-lagr-e05
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.v2_job1.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.v2_job1.%j.txt

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
# Job 1: Embedding + Lagrangian OD_OV + ent_coef=0.05
# Budgets: [60, 80, 100, 200, 1000000] (tight, data-driven)
# ═══════════════════════════════════════════════════════════════
METHOD="lagrangian_od_ov"
BUDGETS="60,80,100,200,1000000"
TIMESTEPS=5000000
NUM_ENVS=8
DENOM_FACTOR=2
BUDGET_ENCODING="embed"
ENT_COEF=0.05

echo "════════════════════════════════════════════"
echo "  Job 1: Embed + Lagrangian + ent=0.05"
echo "  Method:      $METHOD"
echo "  Budgets:     $BUDGETS"
echo "  Encoding:    $BUDGET_ENCODING"
echo "  Ent coef:    $ENT_COEF"
echo "  Timesteps:   $TIMESTEPS"
echo "  Envs:        $NUM_ENVS"
echo "  Denom:       $DENOM_FACTOR"
echo "════════════════════════════════════════════"

stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train \
    --method "$METHOD" \
    --budgets "$BUDGETS" \
    --budget_encoding "$BUDGET_ENCODING" \
    --ent_coef $ENT_COEF \
    --timesteps $TIMESTEPS \
    --num_envs $NUM_ENVS \
    --denom_factor $DENOM_FACTOR \
    2>&1 | tee -a "$PROJ_ROOT/logs/v2_job1.log"
