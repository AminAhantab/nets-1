#! /bin/bash


if [ $# -ne 4 ]; then
    echo "Usage: $0 <seed> <arch> <dataset> <opt> <lr>"
    exit 1
fi

set -euxo pipefail

# Path to python binary (conda environment)
PYTHON_BIN=/home/$USER/.conda/envs/nets/bin/python

# Output path for results
OUT_PATH="results/prune_network"

# Filenames
INIT_FILENAME="init.pt"
MAG_PRUNED_FILENAME="mag_pruned.pt"
RAND_PRUNED_FILENAME="rand_pruned.pt"

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_EVERY=100
LOG_VAL_EVERY=500 # 0 means no validation other than end of epoch
LOG_TEST_EVERY=0 # 0 means no testing other than end of epoch

# Experiment Configuration
SEED=$1

# Network Configuration
ARCH=$2
DATASET=$3
INIT_DENSITY=1.0
PRUNING_RATE=$4

# Make unique output path
TIMESTAMP=$(date +%s)
OUTPUT_PATH="$OUT_PATH/$ARCH/$DATASET/$TIMESTAMP"
LOG_PATH="$OUTPUT_PATH/${ARCH}_prune.log"
mkdir -p $OUTPUT_PATH

# Timestamped Paths
INIT_MODEL_PATH="$OUTPUT_PATH/$INIT_FILENAME"
MAG_PRUNED_MODEL_PATH="$OUTPUT_PATH/$MAG_PRUNED_FILENAME"
RAND_PRUNED_MODEL_PATH="$OUTPUT_PATH/$RAND_PRUNED_FILENAME"

# Initialise Network
$PYTHON_BIN -m nets_cli init \
    --data=$DATASET \
    --arch=$ARCH \
    --density=$INIT_DENSITY \
    --out_path=$INIT_MODEL_PATH \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH

$PYTHON_BIN -m nets_cli prune \
    --model=$INIT_MODEL_PATH \
    --criterion=magnitude \
    --fraction=$PRUNING_RATE \
    --out_path=$MAG_PRUNED_MODEL_PATH \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH

$PYTHON_BIN -m nets_cli prune \
    --model=$INIT_MODEL_PATH \
    --criterion=random \
    --fraction=$PRUNING_RATE \
    --out_path=$RAND_PRUNED_MODEL_PATH \
    --log_level=$LOG_LEVEL \
    --log_file=$LOG_PATH

echo "Pruning complete. Results saved to $OUTPUT_PATH"