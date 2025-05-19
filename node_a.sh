#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

# this file trains the meta learnt model in folder a

cd /rds/user/jr879/hpc-work/
source anaconda3/bin/activate main
cd a/gfn-backward-tests
python src/graph_building/meta_backward.py
cp -r results results_graph_meta
