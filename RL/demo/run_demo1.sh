#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Demo 1: Single Operator -- Constrained RL + CKKS Execution
#
#  Interactive demo for PFE defense.
#  Run on HPC: sbatch or srun --pty bash run_demo1.sh
#  Run locally: bash run_demo1.sh
# ═══════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$RL_DIR")"

# ── Activate conda (HPC) ───────────────────────────────────────
if [ -f /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate ]; then
    set +u
    source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
    conda activate chehab
    set -u
    echo "[env] Conda environment 'chehab' activated"
elif command -v conda &>/dev/null; then
    set +u
    conda activate chehab 2>/dev/null || true
    set -u
fi

# ── Set paths ──────────────────────────────────────────────────
export PYTHONPATH="${RL_DIR}:${RL_DIR}/pytrs:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_demo"
mkdir -p "$MPLCONFIGDIR"

cd "$RL_DIR"

echo "[env] PROJECT_ROOT = $PROJECT_ROOT"
echo "[env] RL_DIR       = $RL_DIR"
echo "[env] PYTHONPATH   = $PYTHONPATH"
echo ""

# ── Verify prerequisites ──────────────────────────────────────
if [ ! -f "$RL_DIR/rules.txt" ]; then
    echo "ERROR: rules.txt not found in $RL_DIR"
    exit 1
fi

if [ ! -d "$RL_DIR/fhe_rl/trained_models" ]; then
    echo "ERROR: trained_models directory not found"
    exit 1
fi

BUILD_DIR="$PROJECT_ROOT/build/RL/veclang_runner"
if [ ! -f "$BUILD_DIR/veclang_runner" ]; then
    echo "WARNING: veclang_runner binary not found at $BUILD_DIR"
    echo "         CKKS compilation will be skipped."
    echo "         Build with: cd $PROJECT_ROOT && cmake -S . -B build && cmake --build build"
fi

# ── Run demo ───────────────────────────────────────────────────
python3 "$SCRIPT_DIR/demo_single_operator.py"
