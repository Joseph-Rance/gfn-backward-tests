#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd d/gfn-backward-test

python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --log-z 16.97 > d_out_0.txt
cp results/models/4999_bck_stop_model.pt backward/4/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/4/base_model.pt
cp results/models/4999_bck_node_model.pt backward/4/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/4/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/5/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/5/base_model.pt
cp results/models/9999_bck_node_model.pt backward/5/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/5/bck_edge_model.pt
cp -r results results_graph_tlm_4
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/4 > d_out_1.txt
cp -r results results_graph_frozen_4
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/5 > d_out_2.txt
cp -r results results_graph_frozen_5
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.01 > d_out_3.txt
cp -r results results_graph_mult_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > d_out_4.txt
cp -r results results_graph_mult_12
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > d_out_5.txt
cp -r results results_graph_mult_13
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 2.00 > d_out_6.txt
cp -r results results_graph_mult_20
