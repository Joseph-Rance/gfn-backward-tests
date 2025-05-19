#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd i/gfn-backward-test

python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > i_out_0.txt
cp -r results results_graph_const_1
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.5 > i_out_1.txt
cp -r results results_graph_const_2
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.5 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > i_out_2.txt
cp -r results results_graph_const_3
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 2.00 > i_out_3.txt
cp -r results results_graph_mult_6
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 2 > i_out_4.txt
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 2 > i_out_5.txt
cp -r results results_graph_tlm_5
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.05 > i_out_6.txt
cp -r results results_graph_max_loss_0
