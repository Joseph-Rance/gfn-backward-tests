#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd h/gfn-backward-test

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
cp -r results results_graph_max_rand_0
