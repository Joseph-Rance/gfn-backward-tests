#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

# this runs files h.sh, and i.sh in directories h, and i

cd /rds/user/jr879/hpc-work

source anaconda3/bin/activate main

(cd h/gfn-backward-tests; bash h.sh > h_out.txt) & (cd i/gfn-backward-tests; bash i.sh > i_out.txt)
