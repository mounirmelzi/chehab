#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 256GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J rl-bootstrap
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.rl_bootstrap.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.rl_bootstrap.%j.txt

# ═══════════════════════════════════════════════════════════════════════
#  Full RL Pipeline Bootstrap Test
#
#  Deep Horner-form polynomials → RL optimization → CHEHAB compilation
#  → CKKS execution with automatic bootstrap insertion
#
#  horner_poly_deg14: degree-14 poly, depth 13  → 1 bootstrap
#  horner_poly_deg26: degree-26 poly, depth 25  → 2 bootstraps
#
#  256 GB memory for bootstrap key generation (LogN=16)
# ═══════════════════════════════════════════════════════════════════════

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
mkdir -p "$PROJ_ROOT/rl_bootstrap_results"

VECLANG_RUNNER="$PROJ_ROOT/build/RL/veclang_runner/veclang_runner"
LATTIGO_DIR="$PROJ_ROOT/lattigo_backend"
RESULTS_DIR="$PROJ_ROOT/rl_bootstrap_results"

echo "═══════════════════════════════════════════════════════════"
echo "  RL PIPELINE BOOTSTRAP TEST"
echo "  Memory: ${SLURM_MEM_PER_NODE:-256G}"
echo "  CPUs:   ${SLURM_CPUS_PER_TASK:-8}"
echo "═══════════════════════════════════════════════════════════"

echo "==> Rebuilding C++ compiler..."
cmake -S . -B build -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" && cmake --build build -j${SLURM_CPUS_PER_TASK:-4}

if [ ! -f "$VECLANG_RUNNER" ]; then
    echo "ERROR: veclang_runner not found after build"
    exit 1
fi

CSV_FILE="$RESULTS_DIR/rl_bootstrap_timing.csv"
echo "name,rl_cost_before,rl_cost_after,rl_reduction_pct,depth,xdepth,terms,ops_cost,compilation_ms,keygen_ms,bootstrap_keygen_ms,encrypt_ms,eval_ms,decrypt_ms,total_ms,precision_bits,abs_error" > "$CSV_FILE"

# ═══════════════════════════════════════════════════════════════
#  Step 1: RL Optimization
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 1: RL Agent optimizes deep polynomial expressions"
echo "═══════════════════════════════════════════════════════════"

BENCH_FILE="$RESULTS_DIR/bootstrap_benchmarks.txt"
grep -E "horner_poly_deg(14|26)$" "$PROJ_ROOT/RL/fhe_rl/datasets/dnn_benchmarks_scaled.txt" > "$BENCH_FILE"
TOTAL_EXPR=$(wc -l < "$BENCH_FILE")
echo "Benchmark file: $BENCH_FILE ($TOTAL_EXPR expressions)"
cat "$BENCH_FILE" | while IFS= read -r line; do
    NAME=$(echo "$line" | rev | cut -d: -f1 | rev)
    LEN=$(echo "$line" | wc -c)
    echo "  $NAME: $LEN chars"
done

cd "$PROJ_ROOT/RL"

BEST_AGENT_DIR="best_model_model_14385922_margin_barrier"
BEST_AGENT="$PROJ_ROOT/RL/eval/${BEST_AGENT_DIR}/best_model.zip"
METHOD="margin_barrier"
TRAIN_BUDGETS="230,236,369,372,9000000"

if [ ! -f "$BEST_AGENT" ]; then
    echo "WARN: margin_barrier agent not found, trying nato_sc..."
    BEST_AGENT_DIR="best_model_model_14529416_nato_sc"
    BEST_AGENT="$PROJ_ROOT/RL/eval/${BEST_AGENT_DIR}/best_model.zip"
    METHOD="nato_sc"
    TRAIN_BUDGETS="230,236,369,9000000"
fi

RL_OUTPUT="$RESULTS_DIR/rl_bootstrap_optimized.xlsx"

if [ -f "$BEST_AGENT" ]; then
    echo ""
    echo "Using agent: $BEST_AGENT_DIR (method=$METHOD)"
    stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
        --model "$BEST_AGENT" \
        --budgets "236,369" \
        --train_budgets "$TRAIN_BUDGETS" \
        --method "$METHOD" \
        --test_mode v2 \
        --benchmark "$BENCH_FILE" \
        --output "$RL_OUTPUT"
    RL_EXIT=$?
    echo "RL optimization exit code: $RL_EXIT"
    if [ -f "$RL_OUTPUT" ]; then
        echo "RL output saved to: $RL_OUTPUT"
    fi
else
    echo "WARN: No trained agent found — RL step skipped"
    echo "       Expressions will still go through veclang_runner (TRS + CKKS)"
fi

# ═══════════════════════════════════════════════════════════════
#  Step 2: Compile and Execute each expression via full pipeline
# ═══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 2: Compile → Lattigo Go → CKKS Execution"
echo "═══════════════════════════════════════════════════════════"

EXPR_IDX=0
PASS=0
FAIL=0
while IFS= read -r line; do
    EXPR_NAME=$(echo "$line" | rev | cut -d: -f1 | rev)
    EXPR_IDX=$((EXPR_IDX + 1))

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [$EXPR_IDX/$TOTAL_EXPR] $EXPR_NAME — Full RL→CKKS Pipeline"
    echo "════════════════════════════════════════════════════════════"

    rm -f "$PROJ_ROOT/build/RL/veclang_runner/generated_fhe.go"

    cd "$PROJ_ROOT/RL"
    TMPEXPR="$RESULTS_DIR/_tmp_expr_${EXPR_NAME}.txt"
    echo "$line" > "$TMPEXPR"

    echo "  [1/6] Generating vectorized code via VecLang generator..."
    python -m veclang_runner.generator --veclang_expression_file "$TMPEXPR"
    GEN_EXIT=$?
    if [ $GEN_EXIT -ne 0 ]; then
        echo "  ERROR: generator failed (exit $GEN_EXIT)"
        FAIL=$((FAIL + 1))
        continue
    fi

    VEC_SIZE=$(wc -c < veclang_runner/temp/vectorized_code.txt)
    echo "  generator OK, vectorized_code.txt = ${VEC_SIZE} bytes"
    if [ "$VEC_SIZE" -eq 0 ]; then
        echo "  ERROR: generator produced empty vectorized_code.txt"
        FAIL=$((FAIL + 1))
        continue
    fi

    cp veclang_runner/temp/fhe_io_example.txt "$PROJ_ROOT/build/RL/veclang_runner/fhe_io_example.txt"
    cp veclang_runner/temp/vectorized_code.txt "$PROJ_ROOT/build/RL/vectorized_code.txt"
    cp veclang_runner/temp/inputs.txt "$PROJ_ROOT/build/RL/inputs.txt"

    cd "$PROJ_ROOT/build/RL/veclang_runner"
    echo "  [2/6] Compiling to Lattigo Go (TRS + CKKS with bootstrap detection)..."
    COMPILE_START=$(($(date +%s%N)/1000000))
    ./veclang_runner 1 1 1 > "$RESULTS_DIR/compile_${EXPR_NAME}.log" 2>&1
    VECLANG_EXIT=$?
    COMPILE_END=$(($(date +%s%N)/1000000))
    COMPILE_MS=$((COMPILE_END - COMPILE_START))

    if [ $VECLANG_EXIT -ne 0 ] || [ ! -f "generated_fhe.go" ]; then
        echo "  ERROR: veclang_runner failed (exit $VECLANG_EXIT, ${COMPILE_MS}ms)"
        tail -10 "$RESULTS_DIR/compile_${EXPR_NAME}.log"
        FAIL=$((FAIL + 1))
        continue
    fi

    GO_LINES=$(wc -l < "generated_fhe.go")
    echo "  Compilation: ${COMPILE_MS}ms, generated_fhe.go = $GO_LINES lines"

    DEPTH=$(grep -oP 'max: \(\K[0-9]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    XDEPTH=$(grep -oP 'max: \([0-9]+, \K[0-9]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    TERMS=$(grep -oP '\|terms\|: \K[0-9]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    OPS_COST=$(grep -oP 'operations_static_cost: \K[0-9.]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    echo "  Circuit stats: depth=$DEPTH, xdepth=$XDEPTH, terms=$TERMS, ops_cost=$OPS_COST"

    echo ""
    echo "  [3/6] Inspecting generated Go code for bootstrap..."
    if grep -q "bootstrapper.Bootstrap" "generated_fhe.go"; then
        BOOT_COUNT=$(grep -c "bootstrapper.Bootstrap" "generated_fhe.go")
        echo "  >>> Found $BOOT_COUNT bootstrap call(s) in generated code <<<"
    else
        echo "  WARNING: No bootstrap calls found!"
    fi

    if grep -q "btpParamsLit" "generated_fhe.go"; then
        echo "  Bootstrap parameter setup: present"
    fi

    echo "  --- Generated params block ---"
    grep -A2 "LogN:" "generated_fhe.go" | head -5
    echo "  --- Bootstrap params ---"
    grep -A5 "btpParamsLit" "generated_fhe.go" | head -8
    echo "  ---"

    echo ""
    echo "  [4/6] Copying to lattigo_backend..."
    cp generated_fhe.go "$LATTIGO_DIR/${EXPR_NAME}.go"
    if [ -f "fhe_io_example_adapted.txt" ]; then
        cp fhe_io_example_adapted.txt "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    else
        rm -f "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    fi
    cd "$LATTIGO_DIR"

    echo "  [5/6] Running instrumented CKKS execution..."
    echo "        This will include bootstrap key generation (LogN=16, memory-intensive)."
    echo "        Memory: $(free -g 2>/dev/null | awk '/^Mem:/{print $3"G used / "$2"G total"}' || echo 'N/A')"

    EXPECTED_VAL=$(python3 "$PROJ_ROOT/scripts/compute_expected.py" veclang "$line" 2>/dev/null || echo "")
    echo "  Expected plaintext value: $EXPECTED_VAL"

    python3 run_instrumented.py "${EXPR_NAME}.go" "$EXPR_NAME" csv "$EXPECTED_VAL" \
        > "$RESULTS_DIR/timing_${EXPR_NAME}.csv" \
        2> "$RESULTS_DIR/instr_${EXPR_NAME}.log"
    INSTR_EXIT=$?

    echo "  [6/6] Results (exit: $INSTR_EXIT)..."
    if [ $INSTR_EXIT -eq 0 ]; then
        DATA_LINE=$(tail -1 "$RESULTS_DIR/timing_${EXPR_NAME}.csv")
        TIMING=$(echo "$DATA_LINE" | cut -d',' -f2-)
        echo "$EXPR_NAME,,,,$DEPTH,$XDEPTH,$TERMS,$OPS_COST,$COMPILE_MS,$TIMING" >> "$CSV_FILE"

        echo ""
        echo "  ═══════════════════════════════════════════════════"
        echo "  BOOTSTRAP TEST PASSED: $EXPR_NAME"
        echo "  $DATA_LINE"
        echo "  ═══════════════════════════════════════════════════"
        PASS=$((PASS + 1))
    else
        echo ""
        echo "  ═══════════════════════════════════════════════════"
        echo "  BOOTSTRAP TEST FAILED: $EXPR_NAME"
        echo "  ═══════════════════════════════════════════════════"
        echo "  --- stderr (last 30 lines) ---"
        grep -v "^\[info\]" "$RESULTS_DIR/instr_${EXPR_NAME}.log" | tail -30
        echo "  ---"
        echo "$EXPR_NAME,,,,$DEPTH,$XDEPTH,$TERMS,$OPS_COST,$COMPILE_MS,0,0,0,0,0,0,0,0" >> "$CSV_FILE"
        FAIL=$((FAIL + 1))
    fi

    cd "$PROJ_ROOT/build/RL/veclang_runner"
done < "$BENCH_FILE"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RL PIPELINE BOOTSTRAP TESTS COMPLETE"
echo "  Passed: $PASS / $TOTAL_EXPR"
echo "  Failed: $FAIL / $TOTAL_EXPR"
echo "  Results: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "CSV contents:"
column -t -s',' "$CSV_FILE"
echo ""
ls -la "$RESULTS_DIR/"
