#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/classification/SEN12MS_EU_B3_moco_LC_rn50_%j.out
#SBATCH --error=srun_outputs/classification/SEN12MS_EU_B3_moco_LC_rn50_%j.err
#SBATCH --time=01:00:00
#SBATCH --job-name=EU_lc_moco
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=develbooster

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000


# load required modules
module load Stages/2022
module load GCCcore/.11.2.0
module load Python

# activate virtual environment
source /p/project/hai_dm4eo/wang_yi/env2/bin/activate


# define available gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3

# run script as slurm job
srun python -u linear_EU_moco.py \
--data_dir /p/project/hai_dm4eo/wang_yi/data/eurosat/tif/ \
--bands RGB \
--checkpoints_dir /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/transfer_classification/checkpoints/SEN12MS_EU_lc_B3_moco_rn50 \
--backbone resnet50 \
--train_frac 1.0 \
--batchsize 64 \
--lr 0.1 \
--schedule 20 40 \
--epochs 50 \
--num_workers 10 \
--seed 42 \
--dist_url $dist_url \
--pretrained /p/project/hai_ssl4eo/wang_yi/SSL4EO-S12/src/benchmark/pretrain_ssl/checkpoints/moco/SEN12MS_B3_rn50_224/checkpoint_0099.pth.tar \
--in_size 224 \
#--normalize \
