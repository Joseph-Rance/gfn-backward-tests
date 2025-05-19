#!/bin/sh
#SBATCH -A COMPUTERLAB-SL2-GPU
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH -p ampere
#SBATCH --gres=gpu:1

# this runs files b.sh, c.sh, d.sh, and e.sh in directories b, c, d, and e

cd /rds/user/jr879/hpc-work

source anaconda3/bin/activate main

(cd b/gfn-backward-tests; bash b.sh) & (cd c/gfn-backward-tests; bash c.sh) & (cd d/gfn-backward-tests; bash d.sh) & (cd e/gfn-backward-tests; bash e.sh)
