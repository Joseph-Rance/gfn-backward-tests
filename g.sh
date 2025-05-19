#!/bin/sh
python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_uniform_3
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --log-z 16.97
cp -r results results_graph_uniform_4
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --log-z 10.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_uniform_5
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_4
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 2.0 --loss-arg-b 3
cp -r results results_graph_biased_1
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --loss-arg-b 3 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_biased_2
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_max_soft_1
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_smooth_0
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.75
cp -r results results_graph_max_smooth_1
