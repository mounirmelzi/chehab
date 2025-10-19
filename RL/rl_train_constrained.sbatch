#!/bin/bash
#SBATCH -J chehab_constr
#SBATCH -o chehab_constr.%j.out
#SBATCH -e chehab_constr.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --reservation=c2
#SBATCH -t 7-00:00:00


# activate env
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u


# live logs
export PYTHONUNBUFFERED=1

# project
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr/chehab"
cd "$PROJ_ROOT"

# ensure imports work (pytrs, etc.)
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"
# quick sanity check: run noise estimator once to verify imports
stdbuf -oL -eL python -u -m RL.pytrs.noise_estimator RL/fhe_rl/datasets/benchmarks.txt 2>&1 | tee -a noise_test.log
# run training via module
stdbuf -oL -eL python -u -m RL.fhe_rl --show_config 2>&1 | tee -a run.log
stdbuf -oL -eL python -u -m RL.fhe_rl train 2>&1 | tee -a run.log