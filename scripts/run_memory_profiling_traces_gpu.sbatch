#!/bin/bash

#SBATCH -p nvidia
#SBATCH -q c2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 32GB
#SBATCH -t 0:30:00

#SBATCH -J mem-traces-gpu
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.mem_traces_gpu.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.mem_traces_gpu.%j.txt

# GPU part of experiment 2(a): real CHEHAB-compiled DNN programs on HEonGPU.

set -uo pipefail

export PROJ_ROOT="/scratch/mxa5343/chehab-constr-updated"
export HEONGPU_DIR="${HOME}/local/heongpu"
MP_DIR="${PROJ_ROOT}/benchmarks/memory_profiling"
OUT_DIR="${PROJ_ROOT}/memory_results"
PROG_DIR="${OUT_DIR}/programs"

module load cuda/12.2.0 2>/dev/null || true
module load gcc 2>/dev/null || true
export LD_LIBRARY_PATH="${HEONGPU_DIR}/lib64:${HEONGPU_DIR}/lib:${HOME}/local/ntl/lib:/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gmp-6.3.0-zims4vx7m6ggtn3ava2r2ksidghaly5v/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

TRACE_GPU="$OUT_DIR/trace_gpu_mem.csv"
echo "name,N,L,K,ops,ctx,skpk,rlk,tools,after_inputs,end,peak" > "$TRACE_GPU"
for P in polynomial_5 polynomial_8 polynomial_10 conv_8_3_1 conv_12_3_2 conv_16_3_2 linear_8_2 linear_16_2 linear_16_3 linear_32_2; do
  for R in 1 2; do
    s=$(date +%s)
    "$MP_DIR/build/gpu_trace_probe" "$PROG_DIR/$P.trace" "$P" >> "$TRACE_GPU" || echo "GPU TRACE FAIL $P"
    echo "[gpu] $P rep $R done in $(( $(date +%s) - s ))s"
  done
done
echo "Finished at $(date)"
