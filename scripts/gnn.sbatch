#!/bin/bash

#SBATCH -q c2
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:3
#SBATCH --time=7-00:00:00
#SBATCH --exclusive

#SBATCH --mail-user=mm15538@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --job-name=gnn_train_gpu
#SBATCH --output=logs/stdout/gnn_train.%j.txt
#SBATCH --error=logs/stderr/gnn_train.%j.txt


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehabEnv

cd RL
stdbuf -oL -eL torchrun --nnodes 1 --nproc_per_node 3 -m fhe_rl.GNNAE train
