#! /bin/bash -l

#SBATCH --job-name=alletf
#SBATCH --chdir=/users/k1502897/workspace/nets/
#SBATCH --partition=cpu
#SBATCH --ntasks=10
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/%j.out

set -euxo pipefail

module load anaconda3/2021.05-gcc-10.3.0

PYTHON_BIN="/scratch/users/k1502897/conda/nets/bin/python"
OUTPUT_DIR="/scratch/users/k1502897/alife/lenet/train_fn"

mkdir -p $OUTPUT_DIR

# Run the experiment
for i in {1..5}; do
    $PYTHON_BIN -m experiments.alife.configurable \
        --trial $i \
        --seed $((42 + $i)) \
        --out_dir $OUTPUT_DIR \
        --arch lenet \
        --dataset mnist \
        --optimiser sgd \
        --lr 1e-3 \
        --max_iter 50000 \
        --fitness 1epoch \
        --cpu_only &
done

wait

echo "All jobs finished successfully"
