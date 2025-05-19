#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd b/gfn-backward-tests

python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 > b_out_0.txt
cp results/models/4999_bck_stop_model.pt backward/0/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/0/base_model.pt
cp results/models/4999_bck_node_model.pt backward/0/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/0/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/1/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/1/base_model.pt
cp results/models/9999_bck_node_model.pt backward/1/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/1/bck_edge_model.pt
cp -r results results_graph_tlm_0
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/0 > b_out_1.txt
cp -r results results_graph_frozen_0
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/1 > b_out_2.txt
cp -r results results_graph_frozen_1
python -u src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.5 > b_out_3.txt
cp -r results results_graph_max_rand_1
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00 > b_out_4.txt
cp -r results results_graph_mult_7
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00 > b_out_5.txt
cp -r results results_graph_mult_8
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 0.10 > b_out_6.txt
cp -r results results_graph_mult_17
