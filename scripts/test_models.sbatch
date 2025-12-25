#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1-00:00:00

#SBATCH -J test_models
#SBATCH -o logs/output.test_models.%j.txt
#SBATCH -e logs/error.test_models.%j.txt

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab

export PROJ_ROOT="/scratch/mxa5343/chehab-constr-picked-agent/chehab"
cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:$PYTHONPATH"

# Test multiple models from training jobs
# Models are saved in: eval/best_model_model_<JOB_ID>/best_model.zip

test_model() {
    local JOB_ID="$1"
    local LABEL="$2"
    local USE_REMAINING_BUDGET="${3:-false}"  # Optional: "true" to enable remaining budget feature
    local MODEL_PATH="$PROJ_ROOT/RL/eval/best_model_model_${JOB_ID}/best_model.zip"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[SKIP] $LABEL: Model not found at $MODEL_PATH"
        return
    fi
    
    echo "[TEST] $LABEL (Job $JOB_ID)"
    echo "  Model: $MODEL_PATH"
    if [ "$USE_REMAINING_BUDGET" = "true" ]; then
        echo "  Using remaining budget feature (257 dims)"
    else
        echo "  Not using remaining budget feature (256 dims)"
    fi
    
    # Run test with this model
    # Override SLURM_JOB_ID so each test gets a unique sheet name
    if [ "$USE_REMAINING_BUDGET" = "true" ]; then
        SLURM_JOB_ID="$LABEL" MODEL_PATH="$MODEL_PATH" stdbuf -oL -eL python -m fhe_rl --tokenizer_type dynamic test --model_path "$MODEL_PATH" --use_remaining_budget
    else
        SLURM_JOB_ID="$LABEL" MODEL_PATH="$MODEL_PATH" stdbuf -oL -eL python -m fhe_rl --tokenizer_type dynamic test --model_path "$MODEL_PATH"
    fi
    
    # Check if test completed successfully
    if [ $? -eq 0 ]; then
        # Rename output file if it exists
        if [ -f "test_results.xlsx" ]; then
            mv -f test_results.xlsx "test_results_${LABEL}.xlsx"
            echo "[OK] Results saved to: test_results_${LABEL}.xlsx"
        else
            echo "[WARN] Test completed but no test_results.xlsx was created"
        fi
    else
        echo "[ERROR] Test failed with exit code $?"
    fi
}

# Test models from your training jobs
# Format: test_model "JOB_ID" "LABEL" "use_remaining_budget" 
# - use_remaining_budget: "true" for models trained WITH remaining budget feature (257 dims)
# - use_remaining_budget: "false" or omitted for old models WITHOUT remaining budget feature (256 dims)

# Model trained WITH remaining budget feature (257 dims)
#test_model "12767812" "train_remaining_budget_300" "true"

# Uncomment to test other models:
# Models trained WITHOUT remaining budget feature (256 dims)
# test_model "12767456" "train_ablation3_always_perstep" "false"
# test_model "12767290" "train_ablation2_always_ondone" "false"
# test_model "12767203" "train_ablation1_perstep_violation" "false"

# Model trained with 9M threshold (256 dims, no remaining budget feature)
test_model "12765829" "train_9mill_budget" "false"

echo ""
echo "[DONE] All tests completed"
ls -lh test_results_*.xlsx 2>/dev/null || echo "No Excel files produced."

