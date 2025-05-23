#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work

source anaconda3/bin/activate main

echo "training template models"

cd a/gfn-backward-tests
python src/graph_building/main.py --loss-fn tb-uniform --seed 1 --save --reward-idx 2 --no-template

cp results/models/9999_base_model.pt results/base_model.pt
cp results/models/9999_fwd_stop_model.pt results/fwd_stop_model.pt
cp results/models/9999_fwd_node_model.pt results/fwd_node_model.pt
cp results/models/9999_fwd_edge_model.pt results/fwd_edge_model.pt

echo "generating template"

python src/graph_building/embeddings.py --save-template --model-path results

cd ../..

echo "duplicating template"

cp a/gfn-backward-tests/results/s/template.npy b/gfn-backward-tests/results/s/template.npy
cp a/gfn-backward-tests/results/s/template.npy c/gfn-backward-tests/results/s/template.npy
cp a/gfn-backward-tests/results/s/template.npy d/gfn-backward-tests/results/s/template.npy

echo "running main scripts"

(cd a/gfn-backward-tests; bash a.sh) & (cd b/gfn-backward-tests; bash b.sh) & (cd c/gfn-backward-tests; bash c.sh) & (cd d/gfn-backward-tests; bash d.sh)
