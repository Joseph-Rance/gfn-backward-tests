#!/bin/sh

#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1
conda activate ~/path/to/project/conda
bash run.sh
