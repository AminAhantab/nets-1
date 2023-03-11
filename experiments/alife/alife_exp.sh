#! /bin/bash -l

#SBATCH --job-name=alife_exp
#SBATCH --chdir=/users/k1502897/workspace/nets/
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/%j.out

set -euxo pipefail

# module load anaconda3/2021.05-gcc-10.3.0

PYTHON_BIN="/home/alexj/.conda/envs/nets/bin/python"
OUTPUT_DIR="/home/alexj/alife_results"

mkdir -p $OUTPUT_DIR

# Run the experiment
for i in {1..5}; do
    $PYTHON_BIN -m experiments.alife.forward_pass --out_dir $OUTPUT_DIR --trial $i
done

echo "Job finished successfully"