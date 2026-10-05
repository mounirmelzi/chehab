#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 128GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J ckks-dnn
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.ckks_dnn.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.ckks_dnn.%j.txt

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

export PATH="/share/apps/NYUAD5/go/1.25.4/bin:$PATH"
export CXX="$SYSTEM_GXX"
export CC="${SYSTEM_GXX/g++/gcc}"
echo "Using C++ compiler: $CXX"

pushd "$PROJ_ROOT/lattigo_backend" > /dev/null
go get golang.org/x/crypto/blake2b@v0.52.0 2>/dev/null || true
go mod tidy 2>/dev/null || true
popd > /dev/null

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/ckks_results"

DEEP_NET="$PROJ_ROOT/build/benchmarks/deep_network/deep_network"
LATTIGO_DIR="$PROJ_ROOT/lattigo_backend"
RESULTS_DIR="$PROJ_ROOT/ckks_results"
RESULTS_CSV="$RESULTS_DIR/ckks_dnn_timing.csv"

echo "═══════════════════════════════════════════════════════════"
echo "  Building C++ compiler"
echo "═══════════════════════════════════════════════════════════"
cmake -S . -B build -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" && cmake --build build -j${SLURM_CPUS_PER_TASK:-4}

if [ ! -f "$DEEP_NET" ]; then
    echo "ERROR: deep_network binary not found"
    exit 1
fi

echo "name,compilation_ms,keygen_ms,bootstrap_keygen_ms,encrypt_ms,eval_ms,decrypt_ms,total_ms,precision_bits,abs_error" > "$RESULTS_CSV"

# ═══════════════════════════════════════════════════════════════
#  Run deep_network C++ benchmarks -> generate Lattigo Go code
#  Then run each generated Go file through instrumented runner
# ═══════════════════════════════════════════════════════════════

PASS=0
FAIL=0
TOTAL=0

compute_expected() {
    local MODE="$1"
    shift
    python3 "$PROJ_ROOT/scripts/compute_expected.py" $MODE "$@" 2>/dev/null
}

run_benchmark() {
    local NAME="$1"
    local EXPECTED_VAL="$2"
    shift 2
    local ARGS="$@"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [$TOTAL] Benchmark: $NAME"
    echo "  Command:   deep_network $ARGS"
    echo "  Expected:  $EXPECTED_VAL"
    echo "════════════════════════════════════════════════════════════"

    cd "$PROJ_ROOT/build/benchmarks/deep_network"

    rm -f generated_*.go

    COMPILE_START=$(($(date +%s%N)/1000000))
    ./deep_network $ARGS 2>&1 | tee "$RESULTS_DIR/${NAME}_compile.log"
    COMPILE_EXIT=$?
    COMPILE_END=$(($(date +%s%N)/1000000))
    COMPILE_MS=$((COMPILE_END - COMPILE_START))

    echo "  Compilation time: ${COMPILE_MS} ms (exit: $COMPILE_EXIT)"

    GO_FILE=$(ls -t generated_*.go 2>/dev/null | head -1)
    if [ -z "$GO_FILE" ] || [ ! -f "$GO_FILE" ]; then
        echo "  SKIPPED — no Go file generated"
        echo "$NAME,$COMPILE_MS,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
        FAIL=$((FAIL + 1))
        return
    fi

    GO_LINES=$(wc -l < "$GO_FILE")
    echo "  Generated: $GO_FILE ($GO_LINES lines)"

    if [ "$GO_LINES" -eq 0 ]; then
        echo "  SKIPPED — generated Go file is empty (0 lines)"
        echo "$NAME,$COMPILE_MS,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
        FAIL=$((FAIL + 1))
        return
    fi

    cp "$GO_FILE" "$LATTIGO_DIR/${NAME}.go"
    # Copy fhe_io_example_adapted.txt if it exists (for plaintext constant values)
    if [ -f "$PROJ_ROOT/build/benchmarks/deep_network/fhe_io_example_adapted.txt" ]; then
        cp "$PROJ_ROOT/build/benchmarks/deep_network/fhe_io_example_adapted.txt" "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    else
        rm -f "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    fi
    cd "$LATTIGO_DIR"

    INSTR_OUTPUT=$(python3 run_instrumented.py "${NAME}.go" "$NAME" csv "$EXPECTED_VAL" 2>"$RESULTS_DIR/${NAME}_instr.log")

    if [ $? -eq 0 ]; then
        DATA_LINE=$(echo "$INSTR_OUTPUT" | tail -1)
        BENCH_NAME=$(echo "$DATA_LINE" | cut -d',' -f1)
        REST=$(echo "$DATA_LINE" | cut -d',' -f2-)
        echo "$BENCH_NAME,$COMPILE_MS,$REST" >> "$RESULTS_CSV"
        echo "  OK: $DATA_LINE"
        PASS=$((PASS + 1))
    else
        echo "  WARN: Instrumented run failed (see ${NAME}_instr.log)"
        tail -3 "$RESULTS_DIR/${NAME}_instr.log"
        echo "$NAME,$COMPILE_MS,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
        FAIL=$((FAIL + 1))
    fi
}

# ═══════════════════════════════════════════════════════════════
#  A. Polynomial circuits (activation function depth scaling)
# ═══════════════════════════════════════════════════════════════

run_benchmark "poly_depth5"   "$(compute_expected polynomial 5)"   polynomial 5
run_benchmark "poly_depth8"   "$(compute_expected polynomial 8)"   polynomial 8
run_benchmark "poly_depth10"  "$(compute_expected polynomial 10)"  polynomial 10
run_benchmark "poly_depth12"  "$(compute_expected polynomial 12)"  polynomial 12

# ═══════════════════════════════════════════════════════════════
#  B. Convolution circuits (CNN feature extractors)
# ═══════════════════════════════════════════════════════════════

run_benchmark "conv_5x5_k2_2layer"   "$(compute_expected conv 5 2 2)"   conv 5 2 2
run_benchmark "conv_8x8_k2_2layer"   "$(compute_expected conv 8 2 2)"   conv 8 2 2
run_benchmark "conv_8x8_k3_1layer"   "$(compute_expected conv 8 3 1)"   conv 8 3 1
run_benchmark "conv_12x12_k3_2layer" "$(compute_expected conv 12 3 2)"  conv 12 3 2
run_benchmark "conv_16x16_k3_2layer" "$(compute_expected conv 16 3 2)"  conv 16 3 2

# ═══════════════════════════════════════════════════════════════
#  C. Linear / FC circuits (MLP layers)
# ═══════════════════════════════════════════════════════════════

run_benchmark "linear_4x4_2layer"   "$(compute_expected linear 4 2)"   linear 4 2
run_benchmark "linear_8x8_2layer"   "$(compute_expected linear 8 2)"   linear 8 2
run_benchmark "linear_16x16_2layer" "$(compute_expected linear 16 2)"  linear 16 2
run_benchmark "linear_16x16_3layer" "$(compute_expected linear 16 3)"  linear 16 3
run_benchmark "linear_32x32_2layer" "$(compute_expected linear 32 2)"  linear 32 2

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL CKKS DNN BENCHMARKS COMPLETE"
echo "  Passed: $PASS / $TOTAL"
echo "  Failed: $FAIL / $TOTAL"
echo "  Results CSV: $RESULTS_CSV"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "CSV contents:"
column -t -s',' "$RESULTS_CSV"
