#!/bin/bash
#SBATCH -J chehab_constr_ppo
#SBATCH -o chehab_constr_ppo.%j.out
#SBATCH -e chehab_constr_ppo.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --reservation=c2
#SBATCH -t 7-00:00:00


# activate env
set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u


# live logs
export PYTHONUNBUFFERED=1

# project
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr/chehab"
cd "$PROJ_ROOT"

# ensure imports work (pytrs, etc.)
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

# run PPO training directly (bypasses config's algorithm switch)
stdbuf -oL -eL python -u -c 'from RL.fhe_rl.__main__ import load_embeddings_from_config, DATA_DIR; from RL.fhe_rl.train import train_ppo_agent; emb,_=load_embeddings_from_config(); train_ppo_agent(str(DATA_DIR / "benchmarks.txt"), emb, total_timesteps=200000)' 2>&1 | tee -a run_ppo.log


