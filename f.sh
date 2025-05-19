#!/bin/sh
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2
cp -r results results_graph_uniform_0
python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 2
cp -r results results_graph_uniform_1
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_uniform_2
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_3
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_16
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.2 --loss-arg-b 3
cp -r results results_graph_biased_0
python src/graph_building/main.py --loss-fn tb-max-ent --seed 1 --save --reward-idx 2
cp -r results results_graph_max_ent_0
python src/graph_building/main.py --loss-fn tb-max-ent --seed 2 --save --reward-idx 2
cp -r results results_graph_max_ent_1
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_soft_0
