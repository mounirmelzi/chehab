#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 4GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0-04:00:00

#SBATCH -J tb-ablation
#SBATCH -o /scratch/mxa5343/chehab-constr-updated/logs/output.tb_ablation.%j.txt
#SBATCH -e /scratch/mxa5343/chehab-constr-updated/logs/error.tb_ablation.%j.txt

source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
module load tensorboard

cd /scratch/mxa5343/chehab-constr-updated/RL/tensorboard

stdbuf -oL -eL tensorboard --logdir_spec="\
V5_LagPID:./model_14529416_nato_sc,\
V5_FOCOPS:./model_14544615_nato_sc,\
V5_PPO_A:./model_14529413_nato_sc,\
V4_MB_B:./model_14385920_margin_barrier,\
V4_MB_FiLM_B:./model_14385922_margin_barrier,\
V4_NMask_B:./model_14385924_noise_masking" \
--host 0.0.0.0 --port 6009
