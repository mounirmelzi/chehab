#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J abl1-unc-safe
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.abl1_unc_safety.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.abl1_unc_safety.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

mkdir -p "$PROJ_ROOT/RL/test_results"

# Ablation 1: Unconstrained PPO (Bilel/Raouf) WITH Safety Rollback
# Purpose: prove that adding safety to an unconstrained agent is NOT as good
#          as a budget-aware agent + safety
echo "════════════════════════════════════════════════════════════"
echo "  ABLATION 1: Unconstrained PPO + Safety Rollback"
echo "════════════════════════════════════════════════════════════"

stdbuf -oL -eL python -u "$PROJ_ROOT/scripts/test_unconstrained_safety.py"

echo "DONE"
