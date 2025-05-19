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
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2
cp -r results results_graph_uniform_0
python src/graph_building/main.py --loss-fn tb-uniform --seed 2 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_uniform_3
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_tlm_2
python src/graph_building/main.py --loss-fn tb-const --seed 1 --save --reward-idx 0 --loss-arg-a 0.2 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_const_1
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_rand_1
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 0 --loss-arg-b 0.10
cp -r results results_graph_mult_2
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 3 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.10
cp -r results results_graph_mult_5
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 2 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 2.00
cp -r results results_graph_mult_8
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 1 --loss-arg-b 0.01 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_12
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 0 --loss-arg-a 0 --loss-arg-b 2.00 --depth 2 --num-features 8 --learning-rate 0.0001 --batch-size 64
cp -r results results_graph_mult_15
python src/graph_building/main.py --loss-fn tb-biased-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 2.0 --loss-arg-b 3
cp -r results results_graph_biased_1
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0
cp -r results results_graph_aligned_0
