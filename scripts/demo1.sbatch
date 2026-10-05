#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J demo1-rl-ckks
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.demo1.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.demo1.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
export RL_DIR="$PROJ_ROOT/RL"
export PYTHONPATH="${RL_DIR}:${RL_DIR}/pytrs:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_demo_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$PROJ_ROOT/logs"

cd "$RL_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  Demo 1: Constrained RL + CKKS Execution"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Pre-selected options (non-interactive) ─────────────────────
#   Operator 5 = poly_activation_deg2_4  (small, fast, depth 1, no bootstrap)
#   Budget   3 = 369 bits (relaxed, high success rate)
#
# Why these choices:
#   - poly_activation_deg2_4 is shallow (no bootstrap), fast to optimize (~1-2s),
#     and produces good precision (26+ bits). Small expression = fewer RL steps.
#   - Budget 369 is the most relaxed constrained budget. The agent can explore
#     more freely, so cost reduction is high and violations are rare.
#     Safety rollback is still likely to trigger (demonstrating the feature).
#
# Alternatives if you want to show a more impressive operator:
#   --operator 1 --budget 3   (conv2d_5x5_k3, B=369 -- larger, 508ms exec)
#   --operator 2 --budget 3   (matmul_4x4, B=369 -- 31.2 bits precision)

python3 "$RL_DIR/demo/demo_single_operator.py" --operator 5 --budget 3

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Demo 1 Complete"
echo "  Trajectory plot: $RL_DIR/demo_trajectory.png"
echo "  Log: $PROJ_ROOT/logs/output.demo1.${SLURM_JOB_ID}.txt"
echo "═══════════════════════════════════════════════════════════"
