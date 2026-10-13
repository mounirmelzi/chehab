#!/bin/bash

#SBATCH -p nvidia
#SBATCH -q c2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 64GB
#SBATCH -t 1:00:00

#SBATCH -J mem-profiling
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.mem_profiling.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.mem_profiling.%j.txt

set -euo pipefail

export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
export HEONGPU_DIR="${HOME}/local/heongpu"
MP_DIR="${PROJ_ROOT}/benchmarks/memory_profiling"
OUT_DIR="${PROJ_ROOT}/memory_results"
REPS=3
mkdir -p "$OUT_DIR"

module load cuda/12.2.0 2>/dev/null || module load cuda 2>/dev/null || true
module load cmake 2>/dev/null || true
module load gcc 2>/dev/null || true
module load go/1.25.4

nvidia-smi --query-gpu=name,memory.total --format=csv

export LD_LIBRARY_PATH="${HEONGPU_DIR}/lib64:${HEONGPU_DIR}/lib:${HOME}/local/ntl/lib:/share/apps/NYUAD6/spack/spack-0.23.0/opt/spack/linux-rocky8-zen/gcc-8.5.0/gmp-6.3.0-zims4vx7m6ggtn3ava2r2ksidghaly5v/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$MP_DIR" -B "$MP_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$MP_DIR/build" -j8

cd "$MP_DIR/cpu_mem_probe"
export GOMODCACHE="$HOME/go/pkg/mod" CGO_ENABLED=0 GOFLAGS=-buildvcs=false
go build -o cpu_mem_probe .

GPU_CSV="$OUT_DIR/gpu_mem.csv"
CPU_CSV="$OUT_DIR/cpu_mem.csv"
echo "backend,N,L,P,k,w,rep,ctx,skpk,rlk,gk,tools,cts,end,peak,driver_used_delta" > "$GPU_CSV"
echo "backend,N,L,P,k,w,rep,ctx,skpk,rlk,gk,tools,cts,end,peak_rss,exact_rlk,exact_gk,exact_ct,exact_pk,exact_sk" > "$CPU_CSV"

while read -r N L; do
  for K in 0 4 16; do
    for W in 1 8; do
      for R in $(seq 1 $REPS); do
        "$MP_DIR/build/gpu_mem_probe" "$N" "$L" "$K" "$W" "$R" >> "$GPU_CSV" || echo "GPU FAIL $N $L $K $W $R"
        "$MP_DIR/cpu_mem_probe/cpu_mem_probe" "$N" "$L" "$K" "$W" "$R" >> "$CPU_CSV" || echo "CPU FAIL $N $L $K $W $R"
      done
      echo "done N=$N L=$L k=$K w=$W"
    done
  done
done < "$MP_DIR/configs.txt"

echo "Results in $OUT_DIR"
