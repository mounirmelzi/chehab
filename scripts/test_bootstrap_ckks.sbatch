#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 256GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J ckks-bootstrap
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.bootstrap.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.bootstrap.%j.txt

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
RESULTS_CSV="$RESULTS_DIR/ckks_bootstrap_timing.csv"

echo "═══════════════════════════════════════════════════════════"
echo "  BOOTSTRAP TEST — poly_depth13 (requires LogN=16)"
echo "  Memory: ${SLURM_MEM_PER_NODE:-256G}"
echo "  CPUs:   ${SLURM_CPUS_PER_TASK:-8}"
echo "═══════════════════════════════════════════════════════════"

echo "==> Rebuilding C++ compiler..."
cmake -S . -B build -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" && cmake --build build -j${SLURM_CPUS_PER_TASK:-4}

if [ ! -f "$DEEP_NET" ]; then
    echo "ERROR: deep_network binary not found"
    exit 1
fi

echo "name,compilation_ms,keygen_ms,bootstrap_keygen_ms,encrypt_ms,eval_ms,decrypt_ms,total_ms,precision_bits,abs_error" > "$RESULTS_CSV"

compute_expected() {
    local MODE="$1"
    shift
    python3 "$PROJ_ROOT/scripts/compute_expected.py" $MODE "$@" 2>/dev/null
}

run_bootstrap_benchmark() {
    local NAME="$1"
    local EXPECTED_VAL="$2"
    shift 2
    local ARGS="$@"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Benchmark: $NAME"
    echo "  Command:   deep_network $ARGS"
    echo "  Expected:  $EXPECTED_VAL"
    echo "════════════════════════════════════════════════════════════"

    cd "$PROJ_ROOT/build/benchmarks/deep_network"
    rm -f generated_*.go

    echo "  [1/5] Compiling circuit..."
    COMPILE_START=$(($(date +%s%N)/1000000))
    ./deep_network $ARGS > "$RESULTS_DIR/${NAME}_compile.log" 2>&1
    COMPILE_EXIT=$?
    COMPILE_END=$(($(date +%s%N)/1000000))
    COMPILE_MS=$((COMPILE_END - COMPILE_START))
    echo "  Compilation: ${COMPILE_MS} ms (exit: $COMPILE_EXIT)"
    cat "$RESULTS_DIR/${NAME}_compile.log"

    GO_FILE=$(ls -t generated_*.go 2>/dev/null | head -1)
    if [ -z "$GO_FILE" ] || [ ! -f "$GO_FILE" ]; then
        echo "  FAILED — no Go file generated"
        return 1
    fi

    GO_LINES=$(wc -l < "$GO_FILE")
    echo "  Generated: $GO_FILE ($GO_LINES lines)"

    if [ "$GO_LINES" -eq 0 ]; then
        echo "  FAILED — empty Go file"
        return 1
    fi

    echo ""
    echo "  [2/5] Inspecting generated Go code for bootstrap..."
    if grep -q "bootstrapper.Bootstrap" "$GO_FILE"; then
        BOOT_COUNT=$(grep -c "bootstrapper.Bootstrap" "$GO_FILE")
        echo "  ✓ Found $BOOT_COUNT bootstrap call(s) in generated code"
    else
        echo "  ✗ WARNING: No bootstrap calls found!"
    fi

    if grep -q "var bootstrapper" "$GO_FILE"; then
        echo "  ✓ Global bootstrapper variable declared"
    else
        echo "  ✗ WARNING: No global bootstrapper variable!"
    fi

    if grep -q "btpParamsLit" "$GO_FILE"; then
        echo "  ✓ Bootstrap parameter setup present"
    else
        echo "  ✗ WARNING: No bootstrap parameter setup!"
    fi

    echo ""
    echo "  --- Generated params block ---"
    grep -A2 "LogN:" "$GO_FILE" | head -5
    echo "  --- Bootstrap params ---"
    grep -A5 "btpParamsLit" "$GO_FILE" | head -8
    echo "  ---"

    echo ""
    echo "  [3/5] Copying to lattigo_backend..."
    cp "$GO_FILE" "$LATTIGO_DIR/${NAME}.go"
    if [ -f "$PROJ_ROOT/build/benchmarks/deep_network/fhe_io_example_adapted.txt" ]; then
        cp "$PROJ_ROOT/build/benchmarks/deep_network/fhe_io_example_adapted.txt" "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    else
        rm -f "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    fi
    cd "$LATTIGO_DIR"

    echo "  [4/5] Running instrumented CKKS execution (this may take a while)..."
    echo "        Bootstrap key generation with LogN=16 is memory-intensive."
    echo "        Memory before run: $(free -g 2>/dev/null | awk '/^Mem:/{print $3"G used / "$2"G total"}' || echo 'N/A')"

    INSTR_OUTPUT=$(python3 run_instrumented.py "${NAME}.go" "$NAME" csv "$EXPECTED_VAL" 2>"$RESULTS_DIR/${NAME}_instr.log")
    EXIT_CODE=$?

    echo "  [5/5] Checking results (exit: $EXIT_CODE)..."

    if [ $EXIT_CODE -eq 0 ]; then
        DATA_LINE=$(echo "$INSTR_OUTPUT" | tail -1)
        BENCH_NAME=$(echo "$DATA_LINE" | cut -d',' -f1)
        REST=$(echo "$DATA_LINE" | cut -d',' -f2-)
        echo "$BENCH_NAME,$COMPILE_MS,$REST" >> "$RESULTS_CSV"
        echo ""
        echo "  ═══════════════════════════════════"
        echo "  BOOTSTRAP TEST PASSED"
        echo "  $DATA_LINE"
        echo "  ═══════════════════════════════════"
    else
        echo ""
        echo "  ═══════════════════════════════════"
        echo "  BOOTSTRAP TEST FAILED"
        echo "  ═══════════════════════════════════"
        echo ""
        echo "  --- Instrumented runner stderr (last 30 lines) ---"
        grep -v "^\[info\]" "$RESULTS_DIR/${NAME}_instr.log" | tail -30
        echo "  ---"
        echo ""
        echo "  --- Full stderr ---"
        cat "$RESULTS_DIR/${NAME}_instr.log"
        echo "  ---"
        echo "$NAME,$COMPILE_MS,0,0,0,0,0,0,0,0" >> "$RESULTS_CSV"
    fi
}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  TEST 1: poly_depth13 — x^(2^13) = x^8192"
echo "  Circuit depth: 13 (exceeds threshold of 12)"
echo "  Expected: bootstrap insertion, LogN=16"
echo "═══════════════════════════════════════════════════════════"

run_bootstrap_benchmark "poly_depth13" "$(compute_expected polynomial 13)" polynomial 13

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  TEST 2: poly_depth25 — x^(2^25) = x^33554432"
echo "  Circuit depth: 25 (requires 2 bootstraps)"
echo "  Expected: 2 bootstrap insertions, LogN=16"
echo "═══════════════════════════════════════════════════════════"

run_bootstrap_benchmark "poly_depth25" "$(compute_expected polynomial 25)" polynomial 25

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  BOOTSTRAP TESTS COMPLETE"
echo "  Results: $RESULTS_CSV"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "CSV contents:"
column -t -s',' "$RESULTS_CSV"
