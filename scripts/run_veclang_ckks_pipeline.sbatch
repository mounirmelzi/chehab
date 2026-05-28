#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 1-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J e2e-pipe
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.e2e_pipeline.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.e2e_pipeline.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

module load go 2>/dev/null || true

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/e2e_results"

VECLANG_RUNNER="$PROJ_ROOT/build/RL/veclang_runner/veclang_runner"
LATTIGO_DIR="$PROJ_ROOT/lattigo_backend"
RESULTS_DIR="$PROJ_ROOT/e2e_results"

# ═══════════════════════════════════════════════════════════════
#  End-to-End Pipeline Test
#  VecLang expr -> RL optimize -> veclang_runner --lattigo ->
#  generated_fhe.go -> instrumented runner -> timing + precision
# ═══════════════════════════════════════════════════════════════

# Build
echo "Building C++ compiler..."
cmake -S . -B build && cmake --build build -j${SLURM_CPUS_PER_TASK:-4}

if [ ! -f "$VECLANG_RUNNER" ]; then
    echo "ERROR: veclang_runner not found"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 1: RL Optimization of DNN expressions"
echo "═══════════════════════════════════════════════════════════"

cd "$PROJ_ROOT/RL"

# Pick the best constrained agent for optimization
BEST_AGENT_DIR="best_model_model_14385922_margin_barrier"
BEST_AGENT="$PROJ_ROOT/RL/eval/${BEST_AGENT_DIR}/best_model.zip"
METHOD="margin_barrier"
TRAIN_BUDGETS="230,236,369,372,9000000"

if [ ! -f "$BEST_AGENT" ]; then
    echo "WARN: Best agent not found at $BEST_AGENT"
    echo "Trying v5 agent..."
    BEST_AGENT_DIR="best_model_model_14529416_nato_sc"
    BEST_AGENT="$PROJ_ROOT/RL/eval/${BEST_AGENT_DIR}/best_model.zip"
    METHOD="nato_sc"
    TRAIN_BUDGETS="230,236,369,9000000"
fi

DNN_BENCH="./fhe_rl/datasets/dnn_benchmarks.txt"
E2E_OUTPUT="$RESULTS_DIR/e2e_rl_optimized.xlsx"

if [ -f "$BEST_AGENT" ]; then
    echo "Using agent: $BEST_AGENT"
    stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
        --model "$BEST_AGENT" \
        --budgets "236,369" \
        --train_budgets "$TRAIN_BUDGETS" \
        --method "$METHOD" \
        --test_mode v2 \
        --benchmark "$DNN_BENCH" \
        --output "$E2E_OUTPUT"
    echo "RL optimization results: $E2E_OUTPUT"
else
    echo "WARN: No trained agent found, skipping RL optimization step"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 2: Compile DNN expressions to Lattigo via veclang_runner"
echo "═══════════════════════════════════════════════════════════"

cd "$PROJ_ROOT/build/RL/veclang_runner"

# Test each DNN expression through the Lattigo pipeline
EXPR_IDX=0
while IFS= read -r line; do
    EXPR_NAME=$(echo "$line" | rev | cut -d: -f1 | rev)
    EXPR_BODY=$(echo "$line" | rev | cut -d: -f2- | rev)
    EXPR_IDX=$((EXPR_IDX + 1))

    echo ""
    echo "─── Expression: $EXPR_NAME ───"

    # Write expression to input files for veclang_runner
    cd "$PROJ_ROOT/RL"
    python -m veclang_runner.generator --veclang_expression_file <(echo "$line")

    cd "$PROJ_ROOT/build/RL/veclang_runner"

    # Run with Lattigo backend (argv[3]=1)
    ./veclang_runner 1 1 1 2>&1 | tee "$RESULTS_DIR/compile_${EXPR_NAME}.log"

    if [ -f "generated_fhe.go" ]; then
        cp generated_fhe.go "$LATTIGO_DIR/generated_e2e_${EXPR_NAME}.go"

        cd "$LATTIGO_DIR"
        echo "Running instrumented benchmark for $EXPR_NAME..."
        python3 run_instrumented.py "generated_e2e_${EXPR_NAME}.go" "$EXPR_NAME" text \
            2>"$RESULTS_DIR/instr_${EXPR_NAME}.log" | tee "$RESULTS_DIR/timing_${EXPR_NAME}.txt"

        cd "$PROJ_ROOT/build/RL/veclang_runner"
    else
        echo "  WARN: No Go file generated for $EXPR_NAME"
    fi
done < "$DNN_BENCH"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  END-TO-END PIPELINE TEST COMPLETE"
echo "  Results in: $RESULTS_DIR/"
echo "═══════════════════════════════════════════════════════════"
ls -la "$RESULTS_DIR"/
