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

#SBATCH -J train_ablation1_perstep_violation
#SBATCH -o /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/output.train_ablation1_perstep_violation.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-picked-agent/chehab/logs/error.train_ablation1_perstep_violation.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-picked-agent/chehab"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# Ablation Study 1: Per-Step Violation Penalty
# - Penalizes on ALL steps when noise > threshold (removed ON_DONE check)
# - Still only penalizes when there's a violation (noise > threshold)
# - Expected: More frequent penalties, agent learns to avoid violations earlier
export LAGRANGIAN_WRAPPER=per_step_violation

# Create logs directory if it doesn't exist (before SLURM tries to write)
mkdir -p "$PROJ_ROOT/logs"

# Run training from RL directory
stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic train 2>&1 | tee -a "$PROJ_ROOT/logs/train_ablation1_perstep_violation.log"

