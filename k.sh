python src/graph_building/main.py --loss-fn tb-max-ent --seed 2 --save --reward-idx 2
cp -r results results_graph_max_ent_1
python src/graph_building/main.py --loss-fn tb-soft-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_soft_0
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.5
cp -r results results_graph_max_smooth_0
python src/graph_building/main.py --loss-fn tb-smooth-tlm --seed 1 --save --reward-idx 2 --loss-arg-a 0.75
cp -r results results_graph_max_smooth_1
python src/graph_building/main.py --loss-fn tb-uniform-rand --seed 1 --save --reward-idx 2 --loss-arg-a 0.25
cp -r results results_graph_max_rand_1
python src/graph_building/main.py --loss-fn tb-loss-aligned --seed 1 --save --reward-idx 2 --loss-arg-a 20 --loss-arg-b 0.1
cp -r results results_graph_max_loss_0
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 3_000
cp -r results results_graph_tlm_10
python src/graph_building/main.py --loss-fn tb-tlm --save --reward-idx 2 --seed 1 --backward-reset-period 5_000
cp -r results results_graph_tlm_11
