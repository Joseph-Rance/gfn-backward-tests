#!/bin/sh

python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 2
cp -r results results_graph_uniform_1
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --log-z 16.97
cp -r results results_graph_uniform_4
python src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_tlm_3
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_const_2
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.01
cp -r results results_graph_mult_0
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_3
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 2.00
cp -r results results_graph_mult_6
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 3 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_10
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_13
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_16
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --loss-arg-b 3 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_biased_2
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 2
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_18
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 4
cp -r results results_graph_tlm_6
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_17
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_20
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --backward-init uniform
cp -r results results_graph_tlm_8
python src/graph_building/main.py --loss-fn tb-free --seed 1 --save --reward-idx 2
cp -r results results_graph_free_0
python src/graph_building/main.py --loss-fn tb-max-ent --seed 1 --save --reward-idx 2
cp -r results results_graph_max_ent_0
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_soft_0
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_smooth_0
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.125
cp -r results results_graph_max_rand_0
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.1
cp -r results results_graph_max_loss_0
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 3_000
cp -r results results_graph_tlm_10
