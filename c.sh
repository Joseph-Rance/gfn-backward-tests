#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd c/gfn-backward-test

python -u src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2 > c_out_0.txt
cp results/models/4999_bck_stop_model.pt backward/2/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/2/base_model.pt
cp results/models/4999_bck_node_model.pt backward/2/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/2/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/3/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/3/base_model.pt
cp results/models/9999_bck_node_model.pt backward/3/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/3/bck_edge_model.pt
cp -r results results_graph_tlm_1
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/2 > c_out_1.txt
cp -r results results_graph_frozen_2
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/3 > c_out_2.txt
cp -r results results_graph_frozen_3
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.01 > c_out_3.txt
cp -r results results_graph_mult_0
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 3 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00 > c_out_4.txt
cp -r results results_graph_mult_10
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > c_out_5.txt
cp -r results results_graph_mult_11
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 4 > c_out_6.txt
cp -r results results_graph_uniform_6
