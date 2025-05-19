#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

# this runs files e.sh, f.sh, and g.sh in directories e, f, and g

cd /rds/user/jr879/hpc-work

source anaconda3/bin/activate main

(cd e/gfn-backward-tests; bash e.sh > e_out.txt) & (cd f/gfn-backward-tests; bash f.sh > f_out.txt) & (cd g/gfn-backward-tests; bash g.sh > g_out.txt)
