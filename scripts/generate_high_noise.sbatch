#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 16GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J gen-high-noise
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.gen_high_noise.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.gen_high_noise.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# OpenRouter API key (set via environment variable before running)
export OPENAI_API_KEY="${OPENROUTER_API_KEY}"

mkdir -p "$PROJ_ROOT/logs"

echo "════════════════════════════════════════════"
echo "  High-Noise Dataset Generation"
echo "  Plan: 6000 expressions, MultDepth 7–12"
echo "  Target: budgets above 220 bits"
echo "════════════════════════════════════════════"

stdbuf -oL -eL python -u "$PROJ_ROOT/scripts/generate_high_noise_dataset.py" \
    --seed 42 \
    --dataset "./fhe_rl/datasets/final_llm_dataset.txt" \
    2>&1 | tee -a "$PROJ_ROOT/logs/gen_high_noise.log"

echo ""
echo "════════════════════════════════════════════"
echo "  Dataset statistics after augmentation:"
echo "════════════════════════════════════════════"
wc -l "./fhe_rl/datasets/final_llm_dataset.txt"
echo "Done."
