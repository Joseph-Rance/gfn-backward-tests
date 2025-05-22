#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work
source anaconda3/bin/activate main
cd new/gfn-backward-tests

python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.019 > m_out_0.txt
cp -r results results_graph_mult_m_0
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.040 > m_out_1.txt
cp -r results results_graph_mult_m_1
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 0.321 > m_out_2.txt
cp -r results results_graph_mult_m_2
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 1.3 > m_out_3.txt
cp -r results results_graph_mult_m_3
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 1.7 > m_out_4.txt
cp -r results results_graph_mult_m_4
python -u src/graph_building/main.py --loss-fn tb-uniform-action-mult --seed 1 --save --reward-idx 2 --loss-arg-a 1 --loss-arg-b 3.0 > m_out_5.txt
cp -r results results_graph_mult_m_5
