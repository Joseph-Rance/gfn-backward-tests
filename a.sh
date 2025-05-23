#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work/
source anaconda3/bin/activate main
cd a/gfn-backward-tests
python -u src/graph_building/meta_backward.py > a_out.txt
cp -r results results_graph_meta
