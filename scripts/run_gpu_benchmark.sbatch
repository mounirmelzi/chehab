#!/bin/bash

#SBATCH -p nvidia
#SBATCH -q c2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 4:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J gpu-benchmark
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.gpu_bench.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.gpu_bench.%j.txt

set -euo pipefail

export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
export HEONGPU_DIR="${HOME}/local/heongpu"

module load cuda/12.2.0 2>/dev/null || module load cuda 2>/dev/null || true
module load cmake 2>/dev/null || true
module load gcc 2>/dev/null || true

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/gpu_results"

echo "═══════════════════════════════════════════════════════════"
echo "  GPU Benchmark (HEonGPU)"
echo "═══════════════════════════════════════════════════════════"
echo ""

nvidia-smi || true
echo ""

# ═══════════════════════════════════════════════════════════════
#  Step 1: Install HEonGPU if not already done
# ═══════════════════════════════════════════════════════════════
if [ ! -d "${HEONGPU_DIR}" ]; then
    echo "HEonGPU not found at ${HEONGPU_DIR}, installing..."
    bash "${PROJ_ROOT}/scripts/setup_heongpu.sh"
else
    echo "HEonGPU already installed at ${HEONGPU_DIR}"
fi

# ═══════════════════════════════════════════════════════════════
#  Step 2: Build standalone GPU benchmark
# ═══════════════════════════════════════════════════════════════
echo ""
echo "Building GPU benchmark..."
BENCH_DIR="${PROJ_ROOT}/benchmarks/gpu_benchmark"
BENCH_BUILD="${BENCH_DIR}/build"

cmake -S "${BENCH_DIR}" -B "${BENCH_BUILD}" \
    -DCMAKE_PREFIX_PATH="${HEONGPU_DIR}" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${BENCH_BUILD}" -j${SLURM_CPUS_PER_TASK:-4}

# ═══════════════════════════════════════════════════════════════
#  Step 3: Run benchmarks
# ═══════════════════════════════════════════════════════════════
export LD_LIBRARY_PATH="${HEONGPU_DIR}/lib64:${HEONGPU_DIR}/lib:${HOME}/local/ntl/lib:/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gmp-6.3.0-zims4vx7m6ggtn3ava2r2ksidghaly5v/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo ""
echo "Running primitive-ops benchmark..."
"${BENCH_BUILD}/gpu_benchmark" 2>&1 | tee "${PROJ_ROOT}/gpu_results/gpu_benchmark_output.txt"

echo ""
echo "Running DNN-operator benchmark..."
"${BENCH_BUILD}/gpu_dnn_benchmark" 2>&1 | tee "${PROJ_ROOT}/gpu_results/gpu_dnn_benchmark_output.txt"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  GPU Benchmarks Complete"
echo "  Results: ${PROJ_ROOT}/gpu_results/"
echo "═══════════════════════════════════════════════════════════"
