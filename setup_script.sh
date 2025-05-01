wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
conda create -n main python=3.10
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install graph-transformer-pytorch
pip install matplotlib
git clone https://github.com/Joseph-Rance/gfn-backward-tests.git
