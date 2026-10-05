#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J demo2-nn-pipeline
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.demo2.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.demo2.%j.txt

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
echo "  Demo 2: Encrypted Neural Network Pipeline"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Demo 2 is fully automated (no interactive prompts).
# It runs 3 layers: matmul_4x4 -> poly_activation_deg2_4 -> fc_plus_act_4
# Each layer: RL optimization + CKKS compile + Lattigo execution
# Budget: 369 (hardcoded in script, relaxed for best results)
# Estimated time: 5-10 minutes total (RL ~1-2s/layer, Lattigo ~0.5-2s/layer)

python3 "$RL_DIR/demo/demo_neural_network.py"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Demo 2 Complete"
echo "  Log: $PROJ_ROOT/logs/output.demo2.${SLURM_JOB_ID}.txt"
echo "═══════════════════════════════════════════════════════════"
