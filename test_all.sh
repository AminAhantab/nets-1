#! /bin/bash

set -euxo pipefail

PYTHON_BIN=/home/$USER/.conda/envs/nets/bin/python

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE="nets.log"

# Experiment Configuration
SEED=42

# Output Configuration
OUT_PATH="results/test_all"

INIT_ARGS="--seed=$SEED --log_level=$LOG_LEVEL --log_file=$LOG_FILE"

$PYTHON_BIN -m nets_cli init --data=mnist --arch=lenet --density=1.0 $INIT_ARGS --out_path=$OUT_PATH/lenet.init.pt
$PYTHON_BIN -m nets_cli init --data=cifar10 --arch=conv-2 --density=1.0 $INIT_ARGS --out_path=$OUT_PATH/conv-2.init.pt
$PYTHON_BIN -m nets_cli init --data=cifar10 --arch=conv-4 --density=1.0 $INIT_ARGS --out_path=$OUT_PATH/conv-4.init.pt
$PYTHON_BIN -m nets_cli init --data=cifar10 --arch=conv-6 --density=1.0 $INIT_ARGS --out_path=$OUT_PATH/conv-6.init.pt

