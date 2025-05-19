#!/bin/sh
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --log-z 16.97
cp results/models/4999_bck_stop_model.pt backward/4/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/4/base_model.pt
cp results/models/4999_bck_node_model.pt backward/4/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/4/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/5/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/5/base_model.pt
cp results/models/9999_bck_node_model.pt backward/5/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/5/bck_edge_model.pt
cp -r results results_graph_tlm_4
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/4
cp -r results results_graph_frozen_4
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/5
cp -r results results_graph_frozen_5
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.01
cp -r results results_graph_mult_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_12
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_13
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_20
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 4 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_aligned_2
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --backward-init uniform
cp -r results results_graph_tlm_8
