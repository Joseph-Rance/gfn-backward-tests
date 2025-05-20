#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd j/gfn-backward-tests

python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.08 > n_out_0.txt
cp -r results results_graph_const_0
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.08 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > n_out_1.txt
cp -r results results_graph_const_1
python -u src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > h_out_5.txt
cp -r results results_graph_aligned_0
python -u src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.125 > h_out_6.txt
cp -r results results_graph_rand_0
python -u src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.05 > i_out_6.txt
cp -r results results_graph_loss_0
python -u src/graph_building/main.py --loss-fn tb-inv-tlm --seed 1 --save --reward-idx 2 > n_out_2.txt
cp -r results results_graph_inv_0
