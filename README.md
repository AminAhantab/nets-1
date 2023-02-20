# Neuroevolution Ticket Search (NeTS)

**Finding Winning Tickets for Deep Neural Networks without Training First**

<p>
    <a href="https://github.com/alexjackson1/nets">
    <div>
        <p>
        <img src="images/DALL·E 2023-02-12 02.29.07.png" alt=""  height="120">
        <img src="images/DALL·E 2023-02-12 02.29.20.png" alt=""  height="120">
        <img src="images/DALL·E 2023-02-12 02.29.25.png" alt=""  height="120">
        <img src="images/DALL·E 2023-02-12 02.29.32.png" alt="" height="120">
        </p>
    </div>
    </a>
</p>

*A DALL-E rendering of sparse, deep neural networks in a print style.*

***

## Getting Started

Assuming you have a working Python and anaconda installation, you can get started with NeTS by following these steps:

1. Clone the repository using `git clone`.
2. Install the dependencies using `conda env create -f environment.yml`.
3. Activate the environment using `conda activate nets`.
4. Run the experiments using `python run_nets.py`.

## The Lottery Ticket Hypothesis

> The *Lottery Ticket Hypothesis* (LTH) asserts that a randomly initialised overparameterised *Deep Neural Network* (DNN) contains a sparse subnetwork that, when trained (up to) the same amount as the original network, performs just as well.

The LTH suggests that resetting an overparameterised DNN to its initialisation (after a period of gradient descent) is, in some way, necessary for finding a highly trainable subnetwork.
*Neuroevolution Ticket Search* (NeTS) is a method for finding a winning ticket for a DNN that uses a genetic algorithm to search for a winning ticket, or initialisation, wihtout first pre-training an overparameterised dense one.
It seeks to emulate the workings of *Iterative Magnitude Pruning* (IMP), introduced alongside the LTH by Frankle and Carbin (2019), by incorporating a fitness signal that includes information both a networks sparsity and trainability.

***This repository contains an implementation of NeTS written in Python that uses PyTorch models to represent phenotypic networks.***

## Experiments

### Experiment One: MLP-10 and MLP-100 on the XOR Problem

The goal of the network in the XOR problem is to learn a correct non-linear decision boundary between two classes of data: true and false.
The XOR problem is a simple example of a non-linearly separable problem, and is often used to test the performance of neural networks.
We use a simple MLP with 10 and 100 hidden units to test the performance of NeTS on a simple problem.
NeTS is able to find a winning ticket for both MLPs in a reasonable amount of time.

### Experiment Two: LeNet-300-100 on the MNIST Character Recognition Dataset

The goal of the network when considering the MNIST dataset is to learn a correct classifier for handwritten digits (arabic numerals from 0 to 9).
We use a feed-forward architecture with 300 and 100 hidden units to test the performance of NeTS on a more complex problem.
NeTS is able to find a winning ticket for LeNet-300-100 in a reasonable amount of time.

## References

Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1903.01611.

***

## Copyright and License

Copyright (c) Alex Jackson, 2023. All rights reserved.

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
