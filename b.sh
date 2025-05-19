#!/bin/sh
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2
cp results/models/4999_bck_stop_model.pt backward/0/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/0/base_model.pt
cp results/models/4999_bck_node_model.pt backward/0/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/0/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/1/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/1/base_model.pt
cp results/models/9999_bck_node_model.pt backward/1/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/1/bck_edge_model.pt
cp -r results results_graph_tlm_0
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/0
cp -r results results_graph_frozen_0
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/1
cp -r results results_graph_frozen_1
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_rand_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_7
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_8
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_17
