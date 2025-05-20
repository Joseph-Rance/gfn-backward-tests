#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd g/gfn-backward-tests

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
