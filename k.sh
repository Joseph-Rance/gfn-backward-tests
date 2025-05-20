#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd k/gfn-backward-tests

python -u src/graph_building/main.py --loss-fn tb-max-ent --seed 2 --save --reward-idx 2 > k_out_0.txt
cp -r results results_graph_max_ent_1
python -u src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5 > k_out_1.txt
cp -r results results_graph_soft_0
python -u src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5 > k_out_2.txt
cp -r results results_graph_smooth_0
python -u src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.75 > k_out_3.txt
cp -r results results_graph_smooth_1
python -u src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.25 > k_out_4.txt
cp -r results results_graph_rand_1
python -u src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.1 > k_out_5.txt
cp -r results results_graph_loss_0
python -u src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 3_000 > k_out_6.txt
cp -r results results_graph_tlm_10
python -u src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 5_000 > k_out_7.txt
cp -r results results_graph_tlm_11
