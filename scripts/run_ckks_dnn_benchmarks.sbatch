#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J ckks-dnn
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.ckks_dnn.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.ckks_dnn.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT"

module load go 2>/dev/null || true

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/ckks_results"

DEEP_NET="$PROJ_ROOT/build/benchmarks/deep_network/deep_network"
LATTIGO_DIR="$PROJ_ROOT/lattigo_backend"
RESULTS_DIR="$PROJ_ROOT/ckks_results"
RESULTS_CSV="$RESULTS_DIR/ckks_dnn_timing.csv"

# Build the C++ compiler
echo "═══════════════════════════════════════════════════════════"
echo "  Building C++ compiler"
echo "═══════════════════════════════════════════════════════════"
cmake -S . -B build && cmake --build build -j${SLURM_CPUS_PER_TASK:-4}

if [ ! -f "$DEEP_NET" ]; then
    echo "ERROR: deep_network binary not found"
    exit 1
fi

# CSV header
echo "name,compilation_ms,keygen_ms,encrypt_ms,eval_ms,bootstrap_ms,decrypt_ms,total_ms,precision_bits,abs_error" > "$RESULTS_CSV"

# ═══════════════════════════════════════════════════════════════
#  Run deep_network C++ benchmarks -> generate Lattigo Go code
#  Then run each generated Go file through instrumented runner
# ═══════════════════════════════════════════════════════════════

run_benchmark() {
    local NAME="$1"
    shift
    local ARGS="$@"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Benchmark: $NAME"
    echo "  Command:   deep_network $ARGS"
    echo "════════════════════════════════════════════════════════════"

    cd "$PROJ_ROOT/build/benchmarks/deep_network"

    COMPILE_START=$(($(date +%s%N)/1000000))
    ./deep_network $ARGS 2>&1 | tee "$RESULTS_DIR/${NAME}_compile.log"
    COMPILE_END=$(($(date +%s%N)/1000000))
    COMPILE_MS=$((COMPILE_END - COMPILE_START))

    echo "  Compilation time: ${COMPILE_MS} ms"

    # Find the generated Go file
    GO_FILE=$(ls -t generated_*.go 2>/dev/null | head -1)
    if [ -z "$GO_FILE" ] || [ ! -f "$GO_FILE" ]; then
        echo "  SKIPPED — no Go file generated"
        echo "$NAME,$COMPILE_MS,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
        return
    fi

    echo "  Generated: $GO_FILE"

    # Copy to lattigo_backend and run instrumented benchmark
    cp "$GO_FILE" "$LATTIGO_DIR/$GO_FILE"
    cd "$LATTIGO_DIR"

    INSTR_OUTPUT=$(python3 run_instrumented.py "$GO_FILE" "$NAME" csv 2>"$RESULTS_DIR/${NAME}_instr.log")

    if [ $? -eq 0 ]; then
        # Extract data line (skip header), prepend compilation time
        DATA_LINE=$(echo "$INSTR_OUTPUT" | tail -1)
        # Insert compilation_ms after name
        BENCH_NAME=$(echo "$DATA_LINE" | cut -d',' -f1)
        REST=$(echo "$DATA_LINE" | cut -d',' -f2-)
        echo "$BENCH_NAME,$COMPILE_MS,$REST" >> "$RESULTS_CSV"
        echo "  Timing recorded."
    else
        echo "  WARN: Instrumented run failed, recording compile time only"
        echo "$NAME,$COMPILE_MS,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
    fi
}

# ─── Polynomial circuits (various depths) ────────────────────
run_benchmark "poly_depth5"  polynomial 5
run_benchmark "poly_depth8"  polynomial 8
run_benchmark "poly_depth10" polynomial 10

# ─── Convolution circuits ────────────────────────────────────
run_benchmark "conv_5x5_k2_2layer" conv 5 2 2
run_benchmark "conv_8x8_k2_2layer" conv 8 2 2
run_benchmark "conv_8x8_k3_1layer" conv 8 3 1

# ─── Linear (FC) circuits ────────────────────────────────────
run_benchmark "linear_4x4_2layer" linear 4 2
run_benchmark "linear_4x4_3layer" linear 4 3


echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL CKKS DNN BENCHMARKS COMPLETE"
echo "  Results CSV: $RESULTS_CSV"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "CSV contents:"
column -t -s',' "$RESULTS_CSV"
