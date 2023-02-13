#!/bin/bash -l

#SBATCH --job-name=xor_mlp
#SBATCH --chdir=/users/k1502897/workspace/nets/
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --signal=USR2
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/users/%u/%j.out

module load anaconda3/2021.05-gcc-10.3.0

/scratch/users/k1502897/conda/nets/bin/python run_nets_seq.py

