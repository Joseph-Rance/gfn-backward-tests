#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

#wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
#bash Anaconda3-2024.10-1-Linux-x86_64.sh
#conda create -n main python=3.10
#pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
#pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
#pip install graph-transformer-pytorch
#pip install matplotlib scikit-learn tqdm pygad
#git clone https://github.com/Joseph-Rance/gfn-backward-tests.git
#conda activate main
#cd gfn-backward-tests

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
