#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

cd /rds/user/jr879/hpc-work

source anaconda3/bin/activate main

(cd b/gfn-backward-tests; bash b.sh > b_out.txt) & (cd c/gfn-backward-tests; bash c.sh > c_out.txt) & (cd d/gfn-backward-tests; bash d.sh > d_out.txt) & (cd e/gfn-backward-tests; bash e.sh > e_out.txt) & (cd f/gfn-backward-tests; bash f.sh > f_out.txt)
