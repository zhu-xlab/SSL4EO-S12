#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/pretrain/B13_train_mae_vits16_70_ep200_%j.out
#SBATCH --error=srun_outputs/pretrain/B13_train_mae_vits16_70_ep200_%j.err
#SBATCH --time=20:00:00
#SBATCH --job-name=pretrain_mae_vits16
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --partition=booster

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
srun python -u pretrain_mae_s2c.py \
--is_slurm_job \
--data_path /p/scratch/hai_ssl4eo/data/ssl4eo_s12/ssl4eo_250k_s2c_uint8.lmdb \
--output_dir /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/mae/B13_vits16_70_ep200 \
--log_dir /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/mae/B13_vits16_70_ep200/log \
--bands B13 \
--model mae_vit_small_patch16 \
--norm_pix_loss \
--mask_ratio 0.7 \
--num_workers 10 \
--batch_size 64 \
--epochs 200 \
--warmup_epochs 10 \
--blr 1.5e-4 \
--weight_decay 0.05 \
--dist_url $dist_url \
--dist_backend 'nccl' \
--seed 42 \
--mode s2c \
--dtype uint8 \
--season random \
--input_size 224 \
#--resume /p/project/hai_dm4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/moco/B13_rn18_int16/checkpoint_0059.pth.tar
