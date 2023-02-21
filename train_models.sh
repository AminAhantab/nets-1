#! /bin/bash

set -euxo pipefail

PYTHON_BIN=/home/$USER/.conda/envs/nets/bin/python

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_FILE="train_nets.log"

# Experiment Configuration
SEED=42
OPTIMISER=adam
LR=0.001
EPOCHS=3
BATCH_SIZE=128
TIMESTAMP=$(date +%s)

# Output Configuration
OUT_PATH="results/train_nets_$TIMESTAMP"

# Initialise Networks
INIT_ARGS="--seed=$SEED --log_level=$LOG_LEVEL --log_file=$LOG_FILE"

$PYTHON_BIN -m nets_cli init \
    --data=mnist \
    --arch=lenet \
    --density=1.0 \
    --out_path=$OUT_PATH/lenet/mnist/init.pt \
    $INIT_ARGS

$PYTHON_BIN -m nets_cli init \
    --data=cifar10 \
    --arch=conv-2 \
    --density=1.0 \
    --out_path=$OUT_PATH/conv-2/cifar10/init.pt \
    $INIT_ARGS

$PYTHON_BIN -m nets_cli init \
    --data=cifar10 \
    --arch=conv-4 \
    --density=1.0 \
    --out_path=$OUT_PATH/conv-4/cifar10/init.pt \
    $INIT_ARGS

# $PYTHON_BIN -m nets_cli init \
#     --data=cifar10 \
#     --arch=conv-6 \
#     --density=1.0 \
#     --out_path=$OUT_PATH/conv-6/cifar10/init.pt \
#     $INIT_ARGS

# Train Networks
TRAIN_ARGS="--seed=$SEED --log_level=$LOG_LEVEL --log_file=$LOG_FILE --epochs=$EPOCHS --batch=$BATCH_SIZE --opt=$OPTIMISER --lr=$LR --no_cuda"

$PYTHON_BIN -m nets_cli train \
    --data=mnist \
    --model=$OUT_PATH/lenet/mnist/init.pt \
    --out_path=$OUT_PATH/lenet/mnist/trained.pt \
    --csv_path=$OUT_PATH/lenet/mnist/training.csv \
    $TRAIN_ARGS

$PYTHON_BIN -m nets_cli train \
    --data=cifar10 \
    --model=$OUT_PATH/conv-2/cifar10/init.pt \
    --out_path=$OUT_PATH/conv-2/cifar10/trained.pt \
    --csv_path=$OUT_PATH/conv-2/cifar10/training.csv \
    $TRAIN_ARGS

$PYTHON_BIN -m nets_cli train \
    --data=cifar10 \
    --model=$OUT_PATH/conv-4/cifar10/init.pt \
    --out_path=$OUT_PATH/conv-4/cifar10/trained.pt \
    --csv_path=$OUT_PATH/conv-4/cifar10/training.csv \
    $TRAIN_ARGS

# $PYTHON_BIN -m nets_cli train \
#     --data=cifar10 \
#     --model=$OUT_PATH/conv-6/cifar10/init.pt \
#     --out_path=$OUT_PATH/conv-6/cifar10/trained.pt \
#     --csv_path=$OUT_PATH/conv-6/cifar10/training.csv \
#     $TRAIN_ARGS
