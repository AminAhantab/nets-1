#! /bin/bash -l

#SBATCH --job-name=nets_exp
#SBATCH --chdir=/users/k1502897/workspace/nets/
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=16G
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/%j.out

set -euxo pipefail

module load anaconda3/2021.05-gcc-10.3.0

PYTHON_BIN="/scratch/users/k1502897/conda/nets/bin/python"
OUTPUT_DIR="/scratch/users/k1502897/nets_vs_rand"

mkdir -p $OUTPUT_DIR

declare -a arr=("conv6_adam_cifar10.py" "conv2_adam_cifar10.py" "conv4_adam_cifar10.py" "lenet_sgd_mnist.py" "lenet_adam_mnist.py")

# Run the experiment
for f in "${arr[@]}"; do
    for i in {1..5}; do
        $PYTHON_BIN experiments/nets_vs_rand/$f --out_dir $OUTPUT_DIR --trial $i
    done
done

echo "Job finished successfully"