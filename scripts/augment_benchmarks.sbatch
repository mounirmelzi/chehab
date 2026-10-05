#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 8GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0-00:30:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J augment-bench
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.augment_bench.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.augment_bench.%j.txt

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

echo "════════════════════════════════════════════"
echo "  Benchmark Augmentation"
echo "  Adding high-noise expressions (MD 7-12)"
echo "  to benchmarks.txt for constrained testing"
echo "════════════════════════════════════════════"

echo ""
echo "Before augmentation:"
wc -l "./fhe_rl/datasets/benchmarks.txt"
echo ""

stdbuf -oL -eL python -u "$PROJ_ROOT/scripts/augment_benchmarks.py" \
    2>&1 | tee -a "$PROJ_ROOT/logs/augment_bench.log"

EXIT_CODE=$?
echo ""
echo "After augmentation:"
wc -l "./fhe_rl/datasets/benchmarks.txt"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "AUGMENTATION SUCCEEDED — benchmarks ready for testing"
else
    echo ""
    echo "AUGMENTATION FAILED — check output above"
fi

exit $EXIT_CODE
