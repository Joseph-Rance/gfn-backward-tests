#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work

source anaconda3/bin/activate main

(cd g/gfn-backward-tests; bash g.sh > g_out.txt) & (cd h/gfn-backward-tests; bash h.sh > h_out.txt) & (cd i/gfn-backward-tests; bash i.sh > i_out.txt) & (cd j/gfn-backward-tests; bash j.sh > j_out.txt) & (cd k/gfn-backward-tests; bash k.sh > k_out.txt)
