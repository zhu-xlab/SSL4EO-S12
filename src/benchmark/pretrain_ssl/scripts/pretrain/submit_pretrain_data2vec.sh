#!/bin/bash -x
#SBATCH --account=hai_ssl4eo
#SBATCH --nodes=1
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=23:30:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4


./srun_train_data2vec_vits16_s2c.sh  