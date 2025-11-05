#!/bin/bash
#SBATCH -J chehab_lag_budget_feature_240_300_1000000
#SBATCH -o chehab_lag_budget_feature_240_300_1000000.%j.out
#SBATCH -e chehab_lag_budget_feature_240_300_1000000.%j.err
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 7-00:00:00

#SBATCH --mail-user=mxa5343@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr/chehab"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# Experiment: Lagrangian PPO with Budget as Feature in Policy
# Method: Budget Multi-Task Learning
# - Budget (240, 300 , 1000000) is encoded as one-hot feature appended to embedding
# - Agent learns 3 underlying policies, one for each budget constraint
# - Budget is randomly selected per episode during training
# - Policy receives 259-dim observation: [256-dim embedding + 3-dim budget one-hot]
# - Agent adapts behavior based on observed budget in the state representation
# - Expected: Agent learns to optimize differently for different noise budgets

stdbuf -oL -eL python -u -m RL.fhe_rl --tokenizer_type dynamic train 2>&1 | tee -a run_lag_budget_feature_240_300_1000000.log
