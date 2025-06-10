#!/bin/sh

python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2
cp -r results results_graph_uniform_0
python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 2
cp -r results results_graph_uniform_1
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_uniform_2
python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_uniform_3
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --log-z 16.97
cp -r results results_graph_uniform_4
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --log-z 10.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_uniform_5
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
python src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2
cp results/models/4999_bck_stop_model.pt backward/2/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/2/base_model.pt
cp results/models/4999_bck_node_model.pt backward/2/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/2/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/3/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/3/base_model.pt
cp results/models/9999_bck_node_model.pt backward/3/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/3/bck_edge_model.pt
cp -r results results_graph_tlm_1
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_tlm_2
python src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_tlm_3
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
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.08
cp -r results results_graph_const_0
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.08 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_const_1
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.2
cp -r results results_graph_const_0
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_const_1
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_const_2
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.5 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_const_3
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.5  # show backward cannot contain too much information
cp -r results results_graph_rand_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.01  # can we encourage certain actions or outputs?
cp -r results results_graph_mult_0
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.01
cp -r results results_graph_mult_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.10
cp -r results results_graph_mult_2
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_3
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_4
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 3 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_5
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 2.00
cp -r results results_graph_mult_6
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_7
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_8
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 3 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_10
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_11
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_12
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_13
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_14
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_15
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_mult_16
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.2 --loss-arg-b 3
cp -r results results_graph_biased_0
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 2.0 --loss-arg-b 3
cp -r results results_graph_biased_1
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --loss-arg-b 3 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_biased_2
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 2.0 --loss-arg-b 3 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_biased_3
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template  # can we encourage a complete solution?
cp -r results results_graph_aligned_0
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 2  # how much connectivity do we need for p_b to matter?
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 2
cp -r results results_graph_tlm_5
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_17
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_18
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 2 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_aligned_1
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 4
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 4
cp -r results results_graph_tlm_6
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_19
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_20
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 4 --learning-rate 0.0001 --batch-size 64 --no-template
cp -r results results_graph_aligned_2
python src/graph_building/main.py --loss-fn tb-tlm --seed 3 --save --reward-idx 2  # generate surface
cp results/models/4999_bck_stop_model.pt backward/6/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/6/base_model.pt
cp results/models/4999_bck_node_model.pt backward/6/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/6/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/7/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/7/base_model.pt
cp results/models/9999_bck_node_model.pt backward/7/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/7/bck_edge_model.pt
cp -r results results_graph_tlm_7
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/0
cp -r results results_graph_frozen_0
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/1
cp -r results results_graph_frozen_1
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/2
cp -r results results_graph_frozen_2
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/3
cp -r results results_graph_frozen_3
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/4
cp -r results results_graph_frozen_4
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/5
cp -r results results_graph_frozen_5
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/6
cp -r results results_graph_frozen_6
python src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/7
cp -r results results_graph_frozen_7
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --backward-init uniform
cp -r results results_graph_tlm_8
python src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2 --backward-init uniform
cp -r results results_graph_tlm_9
python src/graph_building/main.py --loss-fn tb-free --seed 1 --save --reward-idx 2
cp -r results results_graph_free_0
python src/graph_building/main.py --loss-fn tb-free --seed 2 --save --reward-idx 2
cp -r results results_graph_free_1
python src/graph_building/main.py --loss-fn tb-max-ent --seed 1 --save --reward-idx 2
cp -r results results_graph_max_ent_0
python src/graph_building/main.py --loss-fn tb-max-ent --seed 2 --save --reward-idx 2
cp -r results results_graph_max_ent_1
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_soft_0
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_soft_1
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_smooth_0
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.75
cp -r results results_graph_smooth_1
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.125  # solution a: add noise to get out of minima
cp -r results results_graph_rand_0
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_rand_1
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.1  # solution b: loss aligned to directly locate more effective strategies
cp -r results results_graph_loss_0
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.05
cp -r results results_graph_loss_0
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 3_000  # solution c: periodically reset p_b
cp -r results results_graph_tlm_10
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 5_000
cp -r results results_graph_tlm_11

# reruns (seeds deduplicated)
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
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2 > c_out_0.txt
cp results/models/4999_bck_stop_model.pt backward/2/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/2/base_model.pt
cp results/models/4999_bck_node_model.pt backward/2/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/2/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/3/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/3/base_model.pt
cp results/models/9999_bck_node_model.pt backward/3/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/3/bck_edge_model.pt
cp -r results results_graph_tlm_1
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/2 > c_out_1.txt
cp -r results results_graph_frozen_2
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/3 > c_out_2.txt
cp -r results results_graph_frozen_3
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.01 > c_out_3.txt
cp -r results results_graph_mult_0
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 3 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00 > c_out_4.txt
cp -r results results_graph_mult_10
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > c_out_5.txt
cp -r results results_graph_mult_11
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 4 > c_out_6.txt
cp -r results results_graph_uniform_6
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --log-z 16.97 > d_out_0.txt
cp results/models/4999_bck_stop_model.pt backward/4/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/4/base_model.pt
cp results/models/4999_bck_node_model.pt backward/4/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/4/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/5/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/5/base_model.pt
cp results/models/9999_bck_node_model.pt backward/5/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/5/bck_edge_model.pt
cp -r results results_graph_tlm_4
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/4 > d_out_1.txt
cp -r results results_graph_frozen_4
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/5 > d_out_2.txt
cp -r results results_graph_frozen_5
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.01 > d_out_3.txt
cp -r results results_graph_mult_1
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > d_out_4.txt
cp -r results results_graph_mult_12
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > d_out_5.txt
cp -r results results_graph_mult_13
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 2.00 > d_out_6.txt
cp -r results results_graph_mult_20
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 3 --save --reward-idx 2 > e_out_0.txt
cp results/models/4999_bck_stop_model.pt backward/6/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/6/base_model.pt
cp results/models/4999_bck_node_model.pt backward/6/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/6/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/7/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/7/base_model.pt
cp results/models/9999_bck_node_model.pt backward/7/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/7/bck_edge_model.pt
cp -r results results_graph_tlm_7
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/6 > e_out_1.txt
cp -r results results_graph_frozen_6
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/7 > e_out_2.txt
cp -r results results_graph_frozen_7
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.10 > e_out_3.txt
cp -r results results_graph_mult_2
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.10 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > e_out_4.txt
cp -r results results_graph_mult_14
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > e_out_5.txt
cp -r results results_graph_mult_15
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 2 --backward-init uniform > e_out_6.txt
cp -r results results_graph_tlm_9
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 > f_out_0.txt
cp -r results results_graph_uniform_0
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 2 > f_out_1.txt
cp -r results results_graph_uniform_1
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > f_out_2.txt
cp -r results results_graph_uniform_2
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10 > f_out_3.txt
cp -r results results_graph_mult_3
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > f_out_4.txt
cp -r results results_graph_mult_16
python -u src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.2 --loss-arg-b 3 > f_out_5.txt
cp -r results results_graph_biased_0
python -u src/graph_building/main.py --loss-fn tb-max-ent --seed 1 --save --reward-idx 2 > f_out_6.txt
cp -r results results_graph_max_ent_0
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > g_out_0.txt
cp -r results results_graph_uniform_3
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --log-z 16.97 > g_out_1.txt
cp -r results results_graph_uniform_4
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --log-z 10.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > g_out_2.txt
cp -r results results_graph_uniform_5
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10 > g_out_3.txt
cp -r results results_graph_mult_4
python -u src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 2.0 --loss-arg-b 3 > g_out_4.txt
cp -r results results_graph_biased_1
python -u src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --loss-arg-b 3 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > g_out_5.txt
cp -r results results_graph_biased_2
python -u src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.25 > g_out_6.txt
cp -r results results_graph_soft_1
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > h_out_0.txt
cp -r results results_graph_tlm_2
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > h_out_1.txt
cp -r results results_graph_tlm_3
python -u src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.2 > h_out_2.txt
cp -r results results_graph_const_0
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 3 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10 > h_out_3.txt
cp -r results results_graph_mult_5
python -u src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 2.0 --loss-arg-b 3 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > h_out_4.txt
cp -r results results_graph_biased_3
python -u src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > h_out_5.txt
cp -r results results_graph_aligned_0
python -u src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.125 > h_out_6.txt
cp -r results results_graph_rand_0
python -u src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > i_out_0.txt
cp -r results results_graph_const_1
python -u src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.5 > i_out_1.txt
cp -r results results_graph_const_2
python -u src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.5 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > i_out_2.txt
cp -r results results_graph_const_3
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 2.00 > i_out_3.txt
cp -r results results_graph_mult_6
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 2 > i_out_4.txt
cp -r results results_graph_uniform_6
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 2 > i_out_5.txt
cp -r results results_graph_tlm_5
python -u src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.05 > i_out_6.txt
cp -r results results_graph_max_loss_0
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 2.00 > j_out_0.txt
cp -r results results_graph_mult_18
python -u src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 2 --learning-rate 0.0001 --batch-size 64 --no-template > j_out_1.txt
cp -r results results_graph_aligned_1
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 4 > j_out_2.txt
cp -r results results_graph_tlm_6
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 0.10 > j_out_3.txt
cp -r results results_graph_mult_19
python -u src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 4 --learning-rate 0.0001 --batch-size 64 --no-template > j_out_4.txt
cp -r results results_graph_aligned_2
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 > j_out_5.txt
cp -r results results_graph_tlm_8
python -u src/graph_building/main.py --loss-fn tb-free --seed 1 --save --reward-idx 2 > j_out_6.txt
cp -r results results_graph_free_0
python -u src/graph_building/main.py --loss-fn tb-free --seed 2 --save --reward-idx 2 > j_out_7.txt
cp -r results results_graph_free_1
python -u src/graph_building/main.py --loss-fn tb-max-ent --seed 2 --save --reward-idx 2 > k_out_0.txt
cp -r results results_graph_max_ent_1
python -u src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5 > k_out_1.txt
cp -r results results_graph_soft_0
python -u src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5 > k_out_2.txt
cp -r results results_graph_smooth_0
python -u src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.75 > k_out_3.txt
cp -r results results_graph_smooth_1
python -u src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.25 > k_out_4.txt
cp -r results results_graph_rand_1
python -u src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.1 > k_out_5.txt
cp -r results results_graph_loss_0
python -u src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 3_000 > k_out_6.txt
cp -r results results_graph_tlm_10
python -u src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 5_000 > k_out_7.txt
cp -r results results_graph_tlm_11
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.010 --num-batches 4000 > m_out_0.txt
cp -r results results_graph_mult_m_0
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.019 --num-batches 4000 > m_out_1.txt
cp -r results results_graph_mult_m_1
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.040 --num-batches 4000 > m_out_2.txt
cp -r results results_graph_mult_m_2
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.100 --num-batches 4000 > m_out_3.txt
cp -r results results_graph_mult_m_3
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.321 --num-batches 4000 > m_out_4.txt
cp -r results results_graph_mult_m_4
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 1.0 --num-batches 4000 > m_out_5.txt
cp -r results results_graph_mult_m_5
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 1.3 --num-batches 4000 > m_out_6.txt
cp -r results results_graph_mult_m_6
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 1.7 --num-batches 4000 > m_out_7.txt
cp -r results results_graph_mult_m_7
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 3.0 --num-batches 4000 > m_out_8.txt
cp -r results results_graph_mult_m_8
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.001 --num-batches 4000 > m_out_9.txt
cp -r results results_graph_mult_m_9
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.010 --num-batches 4000 > m_out_10.txt
cp -r results results_graph_mult_m_10
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 2 > m_out_11.txt
cp -r results results_graph_mult_m_11
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 5.0 --num-batches 4000 > m_out_12.txt
cp -r results results_graph_mult_m_12
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 10.0 --num-batches 4000 > m_out_13.txt
cp -r results results_graph_mult_m_13
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 30.0 --num-batches 4000 > m_out_14.txt
cp -r results results_graph_mult_m_14 # a
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.001 --num-batches 4000 > m_out_15.txt
cp -r results results_graph_mult_m_15 # b
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.010 --num-batches 4000 > m_out_16.txt
cp -r results results_graph_mult_m_16 # c
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 2 > m_out_17.txt
cp -r results results_graph_mult_m_17 # d
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 5.0 --num-batches 4000 > m_out_18.txt
cp -r results results_graph_mult_m_18 # e
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 10.0 --num-batches 4000 > m_out_19.txt
cp -r results results_graph_mult_m_19 # g
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 30.0 --num-batches 4000 > m_out_20.txt
cp -r results results_graph_mult_m_20 # h
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 > m_out_21.txt  # f
cp results/models/4999_bck_stop_model.pt backward/0/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/0/base_model.pt
cp results/models/4999_bck_node_model.pt backward/0/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/0/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/1/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/1/base_model.pt
cp results/models/9999_bck_node_model.pt backward/1/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/1/bck_edge_model.pt
cp -r results results_graph_mult_m_21
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/0 > m_out_22.txt
cp -r results results_graph_mult_m_22
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/1 > m_out_23.txt
cp -r results results_graph_mult_m_23
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 > m_out_24.txt  # i
cp results/models/4999_bck_stop_model.pt backward/0/bck_stop_model.pt
cp results/models/4999_base_model.pt backward/0/base_model.pt
cp results/models/4999_bck_node_model.pt backward/0/bck_node_model.pt
cp results/models/4999_bck_edge_model.pt backward/0/bck_edge_model.pt
cp results/models/9999_bck_stop_model.pt backward/1/bck_stop_model.pt
cp results/models/9999_base_model.pt backward/1/base_model.pt
cp results/models/9999_bck_node_model.pt backward/1/bck_node_model.pt
cp results/models/9999_bck_edge_model.pt backward/1/bck_edge_model.pt
cp -r results results_graph_mult_m_24
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/0 > m_out_25.txt
cp -r results results_graph_mult_m_25
python -u src/graph_building/main.py --loss-fn tb-frozen --seed 1 --save --reward-idx 2 --backward-init backward/1 > m_out_26.txt
cp -r results results_graph_mult_m_26
python -u src/graph_building/main.py --loss-fn tb-max-ent --seed 1 --save --reward-idx 2 --num-batches 100000 > o_out_0.txt
cp -r results results_graph_big_0
python -u src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --num-batches 100000 > o_out_1.txt
cp -r results results_graph_big_1
python -u src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --num-batches 100000 > o_out_2.txt
cp -r results results_graph_big_2
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.08 > n_out_0.txt
cp -r results results_graph_const_0
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.08 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > n_out_1.txt
cp -r results results_graph_const_1
python -u src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64 --no-template > h_out_5.txt
cp -r results results_graph_aligned_0
python -u src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.125 > h_out_6.txt
cp -r results results_graph_rand_0
python -u src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.05 > i_out_6.txt
cp -r results results_graph_loss_0
python -u src/graph_building/main.py --loss-fn tb-inv-tlm --seed 1 --save --reward-idx 2 > n_out_2.txt
cp -r results results_graph_inv_0
