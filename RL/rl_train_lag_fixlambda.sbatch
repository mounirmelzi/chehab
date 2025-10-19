#!/bin/bash
#SBATCH -J chehab_lag_fixlam
#SBATCH -o chehab_lag_fixlam.%j.out
#SBATCH -e chehab_lag_fixlam.%j.err
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

export FIX_LAMBDA=500
export LAGRANGE_DENOM_FACTOR=1
stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run_lag_fixlambda.log


