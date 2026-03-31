#!/bin/bash

#SBATCH -q c2
#SBATCH -p compute
#SBATCH --mem 32GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 7-00:00:00

#SBATCH -J veclang_generator
#SBATCH -o logs/stdout/veclang_generator.%j.txt
#SBATCH -e logs/stderr/veclang_generator.%j.txt


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehabEnv

cd RL
stdbuf -oL -eL python -m veclang_generator --vec-size 32 --iterations 500
