#!/bin/bash

#SBATCH -p nvidia
#SBATCH -q c2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 120GB
#SBATCH -t 1:00:00

#SBATCH -J mem-profiling-large
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.mem_profiling_large.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.mem_profiling_large.%j.txt

# Experiment 2: (a) real CHEHAB-compiled DNN programs executed as op traces,
#               (b) large synthetic parameters (N up to 65536, up to 64 rotation keys, K = 2, 4).
# Build the probes first (see run_memory_profiling.sbatch); traces come from make_trace.py.

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

PROGRAMS="polynomial_5 polynomial_8 polynomial_10 conv_8_3_1 conv_12_3_2 conv_16_3_2 linear_8_2 linear_16_2 linear_16_3 linear_32_2"

# (a) real programs: CPU runs in the background while the GPU runs in the foreground
TRACE_CPU="$OUT_DIR/trace_cpu_mem.csv"
TRACE_GPU="$OUT_DIR/trace_gpu_mem.csv"
echo "name,N,L,K,ops,setup,after_inputs,end,peak_live,peak_rss" > "$TRACE_CPU"
echo "name,N,L,K,ops,ctx,skpk,rlk,tools,after_inputs,end,peak" > "$TRACE_GPU"
(
  for P in $PROGRAMS; do
    s=$(date +%s)
    "$MP_DIR/cpu_trace_probe/cpu_trace_probe" "$PROG_DIR/$P.trace" "$P" >> "$TRACE_CPU" || echo "CPU TRACE FAIL $P"
    echo "[cpu] $P done in $(( $(date +%s) - s ))s"
  done
) &
CPU_PID=$!
for P in $PROGRAMS; do
  for R in 1 2; do
    s=$(date +%s)
    "$MP_DIR/build/gpu_trace_probe" "$PROG_DIR/$P.trace" "$P" >> "$TRACE_GPU" || echo "GPU TRACE FAIL $P"
    echo "[gpu] $P rep $R done in $(( $(date +%s) - s ))s"
  done
done

# (b) large synthetic configurations: GPU (2 reps), CPU (1 rep)
BIG_GPU="$OUT_DIR/big_gpu_mem.csv"
BIG_CPU="$OUT_DIR/big_cpu_mem.csv"
echo "backend,N,L,P,k,w,rep,ctx,skpk,rlk,gk,tools,cts,end,peak,driver_used_delta" > "$BIG_GPU"
echo "backend,N,L,P,k,w,rep,ctx,skpk,rlk,gk,tools,cts,end,peak_rss,exact_rlk,exact_gk,exact_ct,exact_pk,exact_sk" > "$BIG_CPU"
for CFG in "32768 14 2" "65536 12 2" "65536 20 2" "65536 20 4"; do
  read -r N L K <<< "$CFG"
  for KR in 0 16 64; do
    for W in 1 16; do
      s=$(date +%s)
      for R in 1 2; do
        "$MP_DIR/build/gpu_mem_probe" "$N" "$L" "$KR" "$W" "$R" "$K" >> "$BIG_GPU" || echo "GPU FAIL $N $L $K $KR $W"
      done
      "$MP_DIR/cpu_mem_probe/cpu_mem_probe" "$N" "$L" "$KR" "$W" 1 "$K" >> "$BIG_CPU" || echo "CPU FAIL $N $L $K $KR $W"
      echo "[big] N=$N L=$L K=$K k=$KR w=$W done in $(( $(date +%s) - s ))s"
    done
  done
done

wait $CPU_PID
echo "Finished at $(date). Results in $OUT_DIR"
