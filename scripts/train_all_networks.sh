#! /bin/bash

# scripts/train_network.sh "42" "lenet" "mnist" "adam" "1.2e-3"
scripts/train_network.sh "42" "conv-2" "cifar10" "adam" "2e-4"
scripts/train_network.sh "42" "conv-4" "cifar10" "adam" "3e-4"
scripts/train_network.sh "42" "conv-6" "cifar10" "adam" "3e-4"
