#!/bin/sh
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_const_1
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_const_2
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.5 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_const_3
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 2.00
cp -r results results_graph_mult_6
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 2
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 2
cp -r results results_graph_tlm_5
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.05
cp -r results results_graph_max_loss_0
