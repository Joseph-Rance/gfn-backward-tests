#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd f/gfn-backward-tests

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
