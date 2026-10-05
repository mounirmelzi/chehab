#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 4GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1-00:00:00

#SBATCH -J tensorboard
#SBATCH -o logs/output.tensorboard.txt
#SBATCH -e logs/error.tensorboard.txt


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
module load tensorboard

cd RL/tensorboard
stdbuf -oL -eL tensorboard --logdir ./ --host 0.0.0.0 --port 6006
