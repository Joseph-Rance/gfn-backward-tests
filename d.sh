#!/bin/sh

python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_uniform_2
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --log-z 10.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_uniform_5
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.2
cp -r results results_graph_const_0
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.5 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_const_3
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.01
cp -r results results_graph_mult_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_4
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_7
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_11
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_14
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.2 --loss-arg-b 3
cp -r results results_graph_biased_0
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 2.0 --loss-arg-b 3 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_biased_3
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 2
cp -r results results_graph_tlm_5
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 2 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_aligned_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_19
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 4
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 4 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_aligned_2
python src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2 --backward-init uniform
cp -r results results_graph_tlm_9
python src/graph_building/main.py --loss-fn tb-free --seed 2 --save --reward-idx 2
cp -r results results_graph_free_1
python src/graph_building/main.py --loss-fn tb-max-ent --seed 2 --save --reward-idx 2
cp -r results results_graph_max_ent_1
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_max_soft_1
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.75
cp -r results results_graph_max_smooth_1
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_max_rand_1
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.05
cp -r results results_graph_max_loss_0
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 5_000
cp -r results results_graph_tlm_11
