# baselines
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2
cp -r results results_graph_uniform_0
python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 2
cp -r results results_graph_uniform_1
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --batch-size 1024  # TODO*
cp -r results results_graph_uniform_2
python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_uniform_3
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --log-z 16.97
cp -r results results_graph_uniform_4
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 0 --log-z 10.00 --depth 2 --num-features 8 --batch-size 1024
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
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --batch-size 1024  # TODO*
cp -r results results_graph_tlm_2
python src/graph_building/main.py --loss-fn tb-tlm --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --batch-size 1024
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
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.2
cp -r results results_graph_const_0
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_const_1
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_const_2
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.5 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_const_3

# can we encourage certain actions or outputs?
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.01
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
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.01 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_mult_11
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.01 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_mult_12
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 0.10 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_mult_13
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.10 --depth 2 --num-features 8 --batch-size 1024  # TODO*
cp -r results results_graph_mult_14
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 2.00 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_mult_15
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 2.00 --depth 2 --num-features 8 --batch-size 1024  # TODO*
cp -r results results_graph_mult_16
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.2 --loss-arg-b 3
cp -r results results_graph_biased_0
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 2.0 --loss-arg-b 3
cp -r results results_graph_biased_1
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --loss-arg-b 3 --depth 2 --num-features 8 --batch-size 1024  # TODO*
cp -r results results_graph_biased_2
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 0 --loss-arg-a 2.0 --loss-arg-b 3 --depth 2 --num-features 8 --batch-size 1024
cp -r results results_graph_biased_3

# can we encourage a complete solution?
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0  # TODO*
cp -r results results_graph_aligned_0

# how much connectivity do we need for p_b to matter?
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 2  # TODO *
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 2
cp -r results results_graph_tlm_5
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_17
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_18
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 2 --batch-size 1024  # TODO*
cp -r results results_graph_aligned_1
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --history-bounds 4
cp -r results results_graph_uniform_6
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 4
cp -r results results_graph_tlm_6
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_19
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_20
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 4 --batch-size 1024  # TODO*
cp -r results results_graph_aligned_2

# generate surface
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
python src/graph_building/main.py --loss-fn tb-max-ent --seed 1 --save --reward-idx 2  # TODO*
cp -r results results_graph_max_ent_0
python src/graph_building/main.py --loss-fn tb-max-ent --seed 2 --save --reward-idx 2
cp -r results results_graph_max_ent_1
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5  # TODO*
cp -r results results_graph_max_soft_0
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_max_soft_1
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5  # TODO*
cp -r results results_graph_max_smooth_0
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.75
cp -r results results_graph_max_smooth_1

# solution a: add noise to get out of minima
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.125
cp -r results results_graph_max_rand_0
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_max_rand_1

# solution b: loss aligned to directly locate more effective strategies
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 5 --loss-arg-b 0.5  # TODO*
cp -r results results_graph_max_loss_0

# solution c: periodically reset p_b
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 2_000
cp -r results results_graph_tlm_10
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 3_000  # TODO*
cp -r results results_graph_tlm_11
