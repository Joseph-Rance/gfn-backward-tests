#!/bin/sh

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

source ~/cali/anaconda3/bin/activate main
mkdir results
cd results
mkdir embeddings metrics models batches s m
cd ..
python src/graph_building/embeddings.py --save-template

mkdir backward
cd backward
mkdir 0 1 2 3 4 5 6 7
cd ..
