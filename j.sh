#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd j/gfn-backward-test

python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 2 --loss-arg-a 1 --loss-arg-b 2.00 > j_out_0.txt
cp -r results results_graph_mult_18
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 2 --learning-rate 0.0001 --batch-size 64 --no-template > j_out_1.txt
cp -r results results_graph_aligned_1
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --history-bounds 4 > j_out_2.txt
cp -r results results_graph_tlm_6
python src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --history-bounds 4 --loss-arg-a 1 --loss-arg-b 0.10 > j_out_3.txt
cp -r results results_graph_mult_19
python src/graph_building/main.py --loss-fn tb-aligned --seed 1 --save --reward-idx 0 --history-bounds 4 --learning-rate 0.0001 --batch-size 64 --no-template > j_out_4.txt
cp -r results results_graph_aligned_2
python src/graph_building/main.py --loss-fn tb-tlm --seed 1 --save --reward-idx 2 --backward-init uniform > j_out_5.txt
cp -r results results_graph_tlm_8
python src/graph_building/main.py --loss-fn tb-free --seed 1 --save --reward-idx 2 > j_out_6.txt
cp -r results results_graph_free_0
python src/graph_building/main.py --loss-fn tb-free --seed 2 --save --reward-idx 2 > j_out_7.txt
cp -r results results_graph_free_1
