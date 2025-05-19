#!/bin/sh
python src/graph_building/main.py --loss-fn tb-tlm --seed 3 --save --reward-idx 2
cp results/models/4999_bck_stop_model.pt backward/6/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/6/base_model.pt
cp results/models/4999_bck_node_model.pt backward/6/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/6/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/7/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/7/base_model.pt
cp results/models/9999_bck_node_model.pt backward/7/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/7/bck_edge_model.pt
cp -r results results_graph_tlm_7
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/6
cp -r results results_graph_frozen_6
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/7
cp -r results results_graph_frozen_7
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.10
cp -r results results_graph_mult_2
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_14
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_15
python src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2 --backward-init uniform
cp -r results results_graph_tlm_9
python src/graph_building/main.py --loss-fn tb-free --seed 1 --save --reward-idx 2
cp -r results results_graph_free_0
python src/graph_building/main.py --loss-fn tb-free --seed 2 --save --reward-idx 2
cp -r results results_graph_free_1
