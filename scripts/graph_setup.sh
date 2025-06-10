#!/bin/sh

# cd a; git clone https://github.com/Joseph-Rance/gfn-backward-tests.git; cd gfn-backward-tests; mkdir results; cd results; mkdir embeddings metrics models batches s m; cd ..; mkdir backward; cd backward; mkdir 0 1 2 3 4 5 6 7; cd ..; cp ../../template.npy results/s/template.npy; cd ../..

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
