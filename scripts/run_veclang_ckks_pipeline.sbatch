#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J e2e-pipe
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.e2e_pipeline.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.e2e_pipeline.%j.txt

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
mkdir -p "$PROJ_ROOT/e2e_results"

VECLANG_RUNNER="$PROJ_ROOT/build/RL/veclang_runner/veclang_runner"
LATTIGO_DIR="$PROJ_ROOT/lattigo_backend"
RESULTS_DIR="$PROJ_ROOT/e2e_results"

# ═══════════════════════════════════════════════════════════════
#  End-to-End Pipeline: RL Optimization -> CKKS Compilation -> Execution
#
#  Uses combined DNN benchmarks (original 4 + scaled 7 = 11 expressions)
#  Demonstrates full pipeline: VecLang -> RL agent -> veclang_runner
#    -> Lattigo code generation -> instrumented CKKS execution
# ═══════════════════════════════════════════════════════════════

# Build combined benchmark file
DNN_ORIG="$PROJ_ROOT/RL/fhe_rl/datasets/dnn_benchmarks.txt"
DNN_SCALED="$PROJ_ROOT/RL/fhe_rl/datasets/dnn_benchmarks_scaled.txt"
DNN_BENCH="$RESULTS_DIR/e2e_combined_benchmarks.txt"

cat "$DNN_ORIG" > "$DNN_BENCH"
if [ -f "$DNN_SCALED" ]; then
    cat "$DNN_SCALED" >> "$DNN_BENCH"
fi

TOTAL_EXPR=$(wc -l < "$DNN_BENCH")
echo "═══════════════════════════════════════════════════════════"
echo "  E2E Pipeline: $TOTAL_EXPR DNN expressions"
echo "═══════════════════════════════════════════════════════════"

# Build
echo "Building C++ compiler..."
cmake -S . -B build -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" && cmake --build build -j${SLURM_CPUS_PER_TASK:-4}

if [ ! -f "$VECLANG_RUNNER" ]; then
    echo "ERROR: veclang_runner not found after build"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 1: RL Optimization of DNN expressions"
echo "═══════════════════════════════════════════════════════════"

cd "$PROJ_ROOT/RL"

BEST_AGENT_DIR="best_model_model_14385922_margin_barrier"
BEST_AGENT="$PROJ_ROOT/RL/eval/${BEST_AGENT_DIR}/best_model.zip"
METHOD="margin_barrier"
TRAIN_BUDGETS="230,236,369,372,9000000"

if [ ! -f "$BEST_AGENT" ]; then
    echo "WARN: Best agent not found at $BEST_AGENT"
    echo "Trying nato_sc agent..."
    BEST_AGENT_DIR="best_model_model_14529416_nato_sc"
    BEST_AGENT="$PROJ_ROOT/RL/eval/${BEST_AGENT_DIR}/best_model.zip"
    METHOD="nato_sc"
    TRAIN_BUDGETS="230,236,369,9000000"
fi

E2E_OUTPUT="$RESULTS_DIR/e2e_rl_optimized.xlsx"
RL_OPTIMIZED_EXPRS="$RESULTS_DIR/rl_optimized_expressions.txt"

if [ -f "$BEST_AGENT" ]; then
    echo "Using agent: $BEST_AGENT (method=$METHOD)"
    stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
        --model "$BEST_AGENT" \
        --budgets "236,369" \
        --train_budgets "$TRAIN_BUDGETS" \
        --method "$METHOD" \
        --test_mode v2 \
        --benchmark "$DNN_BENCH" \
        --output "$E2E_OUTPUT" \
        --save_optimized "$RL_OPTIMIZED_EXPRS"
    echo "RL optimization results: $E2E_OUTPUT"
    echo "RL-optimized expressions: $RL_OPTIMIZED_EXPRS"
else
    echo "WARN: No trained agent found, skipping RL optimization step"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 2: Compile RL-optimized expressions to Lattigo & run CKKS"
echo "═══════════════════════════════════════════════════════════"

# Use RL-optimized expressions if available, otherwise fall back to raw
if [ -f "$RL_OPTIMIZED_EXPRS" ] && [ -s "$RL_OPTIMIZED_EXPRS" ]; then
    COMPILE_SRC="$RL_OPTIMIZED_EXPRS"
    echo "Using RL-optimized expressions from: $COMPILE_SRC"
else
    COMPILE_SRC="$DNN_BENCH"
    echo "WARN: No RL-optimized expressions found, using raw expressions"
fi

CSV_FILE="$RESULTS_DIR/e2e_ckks_timing.csv"
echo "expression,depth,xdepth,terms,ops_cost,compile_ms,keygen_ms,bootstrap_keygen_ms,encrypt_ms,eval_ms,decrypt_ms,total_ms,precision_bits,abs_error" > "$CSV_FILE"

COMPILE_TOTAL=$(wc -l < "$COMPILE_SRC")
EXPR_IDX=0
PASS=0
FAIL=0
while IFS= read -r line; do
    EXPR_NAME=$(echo "$line" | rev | cut -d: -f1 | rev)
    EXPR_IDX=$((EXPR_IDX + 1))

    echo ""
    echo "─── [$EXPR_IDX/$COMPILE_TOTAL] Expression: $EXPR_NAME ───"

    rm -f "$PROJ_ROOT/build/RL/veclang_runner/generated_fhe.go"

    cd "$PROJ_ROOT/RL"
    TMPEXPR="$RESULTS_DIR/_tmp_expr_${EXPR_IDX}.txt"
    echo "$line" > "$TMPEXPR"

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
    echo "  compiling to Lattigo..."
    COMPILE_START=$(python3 -c "import time; print(time.perf_counter())")
    ./veclang_runner 1 1 1 > "$RESULTS_DIR/compile_${EXPR_NAME}.log" 2>&1
    VECLANG_EXIT=$?
    COMPILE_END=$(python3 -c "import time; print(time.perf_counter())")
    COMPILE_MS=$(python3 -c "print(round(($COMPILE_END - $COMPILE_START) * 1000, 1))")

    if [ $VECLANG_EXIT -ne 0 ] || [ ! -f "generated_fhe.go" ]; then
        echo "  ERROR: veclang_runner failed (exit $VECLANG_EXIT)"
        tail -5 "$RESULTS_DIR/compile_${EXPR_NAME}.log"
        FAIL=$((FAIL + 1))
        continue
    fi

    DEPTH=$(grep -oP 'max: \(\K[0-9]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    XDEPTH=$(grep -oP 'max: \([0-9]+, \K[0-9]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    TERMS=$(grep -oP '\|terms\|: \K[0-9]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    OPS_COST=$(grep -oP 'operations_static_cost: \K[0-9.]+' "$RESULTS_DIR/compile_${EXPR_NAME}.log" || echo "?")
    echo "  compiled OK: compile=${COMPILE_MS}ms, depth=$DEPTH, xdepth=$XDEPTH, terms=$TERMS, ops_cost=$OPS_COST"

    cp generated_fhe.go "$LATTIGO_DIR/generated_e2e_${EXPR_NAME}.go"
    # Copy fhe_io_example_adapted.txt for plaintext constant values
    if [ -f "fhe_io_example_adapted.txt" ]; then
        cp fhe_io_example_adapted.txt "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    else
        rm -f "$LATTIGO_DIR/fhe_io_example_adapted.txt"
    fi
    cd "$LATTIGO_DIR"
    echo "  running instrumented CKKS benchmark..."

    EXPECTED_VAL=$(python3 "$PROJ_ROOT/scripts/compute_expected.py" veclang "$line" 2>/dev/null || echo "")
    echo "  expected plaintext value: $EXPECTED_VAL"

    python3 run_instrumented.py "generated_e2e_${EXPR_NAME}.go" "$EXPR_NAME" csv $EXPECTED_VAL \
        > "$RESULTS_DIR/timing_${EXPR_NAME}.csv" \
        2> "$RESULTS_DIR/instr_${EXPR_NAME}.log"
    INSTR_EXIT=$?

    if [ $INSTR_EXIT -ne 0 ]; then
        echo "  WARN: instrumented runner failed (exit $INSTR_EXIT)"
        tail -5 "$RESULTS_DIR/instr_${EXPR_NAME}.log"
        echo "$EXPR_NAME,$DEPTH,$XDEPTH,$TERMS,$OPS_COST,$COMPILE_MS,,,,,,,,," >> "$CSV_FILE"
        FAIL=$((FAIL + 1))
    else
        TIMING_DATA=$(tail -1 "$RESULTS_DIR/timing_${EXPR_NAME}.csv" | cut -d',' -f2-)
        echo "$EXPR_NAME,$DEPTH,$XDEPTH,$TERMS,$OPS_COST,$COMPILE_MS,$TIMING_DATA" >> "$CSV_FILE"
        echo "  CKKS execution OK: $TIMING_DATA"
        PASS=$((PASS + 1))
    fi

    cd "$PROJ_ROOT/build/RL/veclang_runner"
done < "$COMPILE_SRC"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 3: Extract RL timing summary"
echo "═══════════════════════════════════════════════════════════"

RL_TIMING_CSV="$RESULTS_DIR/rl_timing_summary.csv"
if [ -f "$E2E_OUTPUT" ]; then
    python3 -c "
import pandas as pd, sys
df = pd.read_excel('$E2E_OUTPUT', sheet_name='all_results')
cols = ['Expression Name', 'Budget', 'RL Time (ms)', 'Initial Cost', 'Safe Final Cost', 'Safe Cost Reduction (%)', 'Safe Final Noise', 'Safety Activated']
out = df[[c for c in cols if c in df.columns]]
out.to_csv('$RL_TIMING_CSV', index=False)
print(f'RL timing summary: {len(out)} rows')
print(out.to_string(index=False))
"
    echo "RL timing CSV: $RL_TIMING_CSV"
else
    echo "WARN: No RL output xlsx found, skipping timing extraction"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  END-TO-END PIPELINE COMPLETE"
echo "  Passed: $PASS / $EXPR_IDX"
echo "  Failed: $FAIL / $EXPR_IDX"
echo "  Results: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Combined CSV:"
column -t -s',' "$CSV_FILE"
echo ""
ls -la "$RESULTS_DIR"/
