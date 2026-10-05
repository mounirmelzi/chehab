#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J validate-rollback
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.validate_rollback.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.validate_rollback.%j.txt

module load gcc 2>/dev/null || true
module load go 2>/dev/null || true
SYSTEM_GXX=$(which g++ 2>/dev/null || echo /usr/bin/g++)

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

export PATH="/share/apps/NYUAD5/go/1.25.4/bin:$PATH"
export CXX="$SYSTEM_GXX"
export CC="${SYSTEM_GXX/g++/gcc}"
echo "Using C++ compiler: $CXX"

pushd "$PROJ_ROOT/lattigo_backend" > /dev/null
go get golang.org/x/crypto/blake2b@v0.52.0 2>/dev/null || true
go mod tidy 2>/dev/null || true
popd > /dev/null

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/validation_results"

# ═══════════════════════════════════════════════════════════════
#  Safety Rollback Cross-Validation
#
#  For each DNN expression:
#    1. Estimate BFV noise using the NNLS noise estimator
#    2. Compile to CKKS/Lattigo via veclang_runner
#    3. Execute and measure actual CKKS precision
#    4. Report whether expressions deemed "safe" by the estimator
#       actually produce correct CKKS results
#
#  This validates the claim that safety rollback guarantees
#  correctness with respect to the noise estimator.
# ═══════════════════════════════════════════════════════════════

echo "Building C++ compiler..."
cmake -S . -B build -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" && cmake --build build -j${SLURM_CPUS_PER_TASK:-4}

if [ ! -f "$PROJ_ROOT/build/RL/veclang_runner/veclang_runner" ]; then
    echo "ERROR: veclang_runner not found after build"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Running cross-validation: noise estimator vs CKKS"
echo "═══════════════════════════════════════════════════════════"
echo ""

cd "$PROJ_ROOT/RL"
python3 "$PROJ_ROOT/scripts/validate_rollback_ckks.py"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  VALIDATION COMPLETE"
echo "  Results: $PROJ_ROOT/validation_results/rollback_validation.csv"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ -f "$PROJ_ROOT/validation_results/rollback_validation.csv" ]; then
    echo "CSV contents:"
    column -t -s',' "$PROJ_ROOT/validation_results/rollback_validation.csv"
fi
