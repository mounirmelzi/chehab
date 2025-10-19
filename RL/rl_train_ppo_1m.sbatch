#!/bin/bash
#SBATCH -J chehab_ppo_1m
#SBATCH -o chehab_ppo_1m.%j.out
#SBATCH -e chehab_ppo_1m.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --reservation=c2
#SBATCH -t 3-00:00:00

set +u
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehab
set -u

export PYTHONUNBUFFERED=1
export NETID=mxa5343
export PROJ_ROOT="/scratch/${NETID}/chehab-constr/chehab"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

stdbuf -oL -eL python -u -c 'from RL.fhe_rl.__main__ import load_embeddings_from_config, DATA_DIR; from RL.fhe_rl.train import train_ppo_agent; emb,_=load_embeddings_from_config(); train_ppo_agent(str(DATA_DIR / "benchmarks.txt"), emb, total_timesteps=1_000_000)' 2>&1 | tee -a run_ppo_1m.log


