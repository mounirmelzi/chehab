#!/bin/bash

#SBATCH --reservation=c2
#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 7-00:00:00

#SBATCH --mail-user=mm15538@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH -J veclang_runner
#SBATCH -o logs/output.veclang_runner.%j.txt
#SBATCH -e logs/error.veclang_runner.%j.txt


source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehabEnv

cd RL
stdbuf -oL -eL python -m veclang_runner
