#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J test-all-models
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.test_all_models.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.test_all_models.%j.txt

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr-updated"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/RL/test_results"

# ═══════════════════════════════════════════════════════════════
# Test ALL trained models on their respective budgets
# Each model tested on the budgets it was trained on
# Results go to test_results/ directory
# ═══════════════════════════════════════════════════════════════

run_test() {
    local JOB_ID="$1"
    local NAME="$2"
    local METHOD="$3"
    local TRAIN_BUDGETS="$4"
    local MODEL_DIR="$5"
    local MODEL_PATH="$PROJ_ROOT/RL/eval/${MODEL_DIR}/best_model.zip"
    local OUTPUT="$PROJ_ROOT/RL/test_results/test_${NAME}.xlsx"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Testing: $NAME (job $JOB_ID)"
    echo "  Method:  $METHOD"
    echo "  Budgets: $TRAIN_BUDGETS"
    echo "  Model:   $MODEL_PATH"
    echo "  Output:  $OUTPUT"
    echo "════════════════════════════════════════════════════════════"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIPPED — model not found (training may not be done yet)"
        return
    fi

    stdbuf -oL -eL python -u -m fhe_rl --tokenizer_type dynamic test \
        --model "$MODEL_PATH" \
        --budgets "$TRAIN_BUDGETS" \
        --train_budgets "$TRAIN_BUDGETS" \
        --method "$METHOD" \
        --output "$OUTPUT"

    echo "  DONE — results at $OUTPUT"
}

# ─── margin_barrier models ────────────────────────────────────
run_test 14085678 "margin_barrier_3b_8envs"    margin_barrier "240,300,9000000" \
    "best_model_model_14085678_margin_barrier"

run_test 14085682 "margin_barrier_5b_8envs"    margin_barrier "300,400,500,1000,9000000" \
    "best_model_model_14085682_margin_barrier"

run_test 14085683 "margin_barrier_10b_8envs"   margin_barrier "240,300,400,500,600,700,800,900,1000,9000000" \
    "best_model_model_14085683_margin_barrier"

# ─── margin_barrier + FiLM ────────────────────────────────────
run_test 14085784 "mb_film_3b_8envs"           margin_barrier "240,300,9000000" \
    "best_model_model_14085784_margin_barrier"

# ─── lagrangian_od_ov models ─────────────────────────────────
run_test 14085679 "lagrangian_od_ov_3b_8envs"  lagrangian_od_ov "240,300,9000000" \
    "best_model_model_14085679_lagrangian_od_ov"

run_test 14085684 "lagrangian_od_ov_5b_8envs"  lagrangian_od_ov "300,400,500,1000,9000000" \
    "best_model_model_14085684_lagrangian_od_ov"

# ─── lagrangian_od_ov + embed ────────────────────────────────
run_test 14085785 "lodov_embed_3b_8envs"       lagrangian_od_ov "240,300,9000000" \
    "best_model_model_14085785_lagrangian_od_ov"

# ─── lagrangian_od_ov 12envs (noise fix) ─────────────────────
run_test 14088790 "lagrangian_od_ov_5b_12envs" lagrangian_od_ov "300,400,500,1000,9000000" \
    "best_model_model_14088790_lagrangian_od_ov"

run_test 14088795 "lodov_embed_3b_12envs"      lagrangian_od_ov "240,300,9000000" \
    "best_model_model_14088795_lagrangian_od_ov"

# ─── lagrangian_always_done models ───────────────────────────
run_test 14085686 "lagrangian_always_done_5b_8envs" lagrangian_always_done "300,400,500,1000,9000000" \
    "best_model_model_14085686_lagrangian_always_done"

# ─── lagrangian_perstep models ───────────────────────────────
run_test 14085688 "lagrangian_perstep_5b_8envs" lagrangian_perstep "300,400,500,1000,9000000" \
    "best_model_model_14085688_lagrangian_perstep"

# ─── old training scripts (before unified codebase) ──────────
run_test 14085644 "old_5budgets_8envs"         lagrangian_od_ov "300,400,500,1000,9000000" \
    "best_model_model_14085644"

run_test 14085645 "old_10budgets_8envs"        lagrangian_od_ov "240,300,400,500,600,700,800,900,1000,9000000" \
    "best_model_model_14085645"

run_test 14085649 "old_5budgets_denom2_8envs"  lagrangian_od_ov "300,400,500,1000,9000000" \
    "best_model_model_14085649"

run_test 14085650 "old_10budgets_denom2_8envs" lagrangian_od_ov "240,300,400,500,600,700,800,900,1000,9000000" \
    "best_model_model_14085650"

# ═══════════════════════════════════════════════════════════════
# V2 models — tight budgets [60,80,100,200,1M] + new methods
# ═══════════════════════════════════════════════════════════════

# ─── v2 Job 1: Embed + Lagrangian + ent=0.05 ─────────────────
run_test 14147263 "v2_embed_lagr_ent05"        lagrangian_od_ov "60,80,100,200,1000000" \
    "best_model_model_14147263_lagrangian_od_ov"

# ─── v2 Job 3: Noise Masking + Embed + ent=0.05 ──────────────
run_test 14147264 "v2_noise_masking"           noise_masking "60,80,100,200,1000000" \
    "best_model_model_14147264_noise_masking"

# ─── v2 Job 4: Curriculum + Lagrangian + Embed + ent=0.05 ────
run_test 14147265 "v2_curriculum"              lagrangian_od_ov "60,80,100,200,1000000" \
    "best_model_model_14147265_lagrangian_od_ov"

# ─── v2 Job 2: Embed + Lagrangian + ent=0.03 ─────────────────
run_test 14147266 "v2_embed_lagr_ent03"        lagrangian_od_ov "60,80,100,200,1000000" \
    "best_model_model_14147266_lagrangian_od_ov"

# ─── v2 Job 5: CONTROL — Embed + Lagrangian + ent=0.0 ────────
run_test 14147267 "v2_embed_noent"             lagrangian_od_ov "60,80,100,200,1000000" \
    "best_model_model_14147267_lagrangian_od_ov"

# ─── v2 Job 6: CONTROL — Raw + Lagrangian + ent=0.05 ─────────
run_test 14147268 "v2_raw_lagr_ent05"          lagrangian_od_ov "60,80,100,200,1000000" \
    "best_model_model_14147268_lagrangian_od_ov"


echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ALL TESTS COMPLETE"
echo "  Results in: $PROJ_ROOT/RL/test_results/"
echo "════════════════════════════════════════════════════════════"
ls -la "$PROJ_ROOT/RL/test_results/"
