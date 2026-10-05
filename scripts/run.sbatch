#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -t 1-00:00:00

#SBATCH -J run
#SBATCH -o logs/output.run.txt
#SBATCH -e logs/error.run.txt


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehabEnv

stdbuf -oL -eL python run_benchmarks.py
