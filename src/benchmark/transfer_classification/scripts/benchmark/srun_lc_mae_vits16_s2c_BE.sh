#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=srun_outputs/classification/BE_mae_LC_vits16_100_%j.out
#SBATCH --error=srun_outputs/classification/BE_mae_LC_vits16_100_%j.err
#SBATCH --time=10:00:00
#SBATCH --job-name=BE_LC_mae_vits16
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
srun python -u linear_BE_mae.py \
--is_slurm_job \
--data_path /p/scratch/hai_ssl4eo/data/bigearthnet/BigEarthNet_LMDB_uint8 \
--output_dir /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/mae_lc/BE_vits16_100 \
--log_dir /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/mae_lc/BE_vits16_100/log \
--model vit_small_patch16 \
--nb_classes 19 \
--train_frac 1.0 \
--num_workers 10 \
--batch_size 64 \
--epochs 100 \
--lr 1.0 \
--warmup_epochs 0 \
--dist_url $dist_url \
--dist_backend 'nccl' \
--seed 42 \
--finetune /p/project/hai_ssl4eo/wang_yi/ssl4eo-s12-dataset/src/benchmark/fullset_temp/checkpoints/mae/B13_vits16_70/checkpoint-99.pth
