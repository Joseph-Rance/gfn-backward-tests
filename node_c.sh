#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

# this runs files f.sh, g.sh, h.sh, and i.sh in directories f, g, h, and i

cd /rds/user/jr879/hpc-work

source anaconda3/bin/activate main

(cd f/gfn-backward-tests; bash f.sh) & (cd g/gfn-backward-tests; bash g.sh) & (cd h/gfn-backward-tests; bash h.sh) & (cd i/gfn-backward-tests; bash i.sh)
