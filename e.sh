#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd e/gfn-backward-test

python -u src/graph_building/main.py --loss-fn tb-tlm --seed 3 --save --reward-idx 2 > e_out_0.txt
cp results/models/4999_bck_stop_model.pt backward/6/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/6/base_model.pt
cp results/models/4999_bck_node_model.pt backward/6/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/6/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/7/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/7/base_model.pt
cp results/models/9999_bck_node_model.pt backward/7/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/7/bck_edge_model.pt
cp -r results results_graph_tlm_7
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/6 > e_out_1.txt
cp -r results results_graph_frozen_6
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/7 > e_out_2.txt
cp -r results results_graph_frozen_7
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.10 > e_out_3.txt
cp -r results results_graph_mult_2
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > e_out_4.txt
cp -r results results_graph_mult_14
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > e_out_5.txt
cp -r results results_graph_mult_15
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2 --backward-init uniform > e_out_6.txt
cp -r results results_graph_tlm_9
