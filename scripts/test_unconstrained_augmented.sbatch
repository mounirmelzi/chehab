#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 16GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-unc-aug
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_unc_aug.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_unc_aug.%j.txt

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
mkdir -p "$PROJ_ROOT/RL/test_results"

echo "════════════════════════════════════════════"
echo "  Testing unconstrained PPO on augmented benchmarks"
echo "  Using standalone script (compatible with old model obs space)"
echo "════════════════════════════════════════════"

N_LINES=$(wc -l < "./fhe_rl/datasets/benchmarks.txt")
echo "Benchmark file: $N_LINES lines"

stdbuf -oL -eL python -u "$PROJ_ROOT/scripts/test_unconstrained_standalone.py" \
    2>&1 | tee -a "$PROJ_ROOT/logs/test_unc_aug.log"
