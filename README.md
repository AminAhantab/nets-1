# Neuroevolution Ticket Search (NeTS)

**Finding Winning Tickets for Deep Neural Networks without Training First**

<div style="text-align:center">
    <p>
        <a href="https://github.com/alexjackson1/nets">
        <div style="display:flex;justify-content:center;margin-top:3rem;margin-bottom:1rem;">
            <p style="display:flex;gap:1rem;">
            <img src="images/DALL·E 2023-02-12 02.29.07.png" alt=""  height="120">
            <img src="images/DALL·E 2023-02-12 02.29.20.png" alt=""  height="120">
            <img src="images/DALL·E 2023-02-12 02.29.25.png" alt=""  height="120">
            <img src="images/DALL·E 2023-02-12 02.29.32.png" alt="" height="120">
            </p>
        </div>
        </a>
    </p>
    <small style="text-align:center;">
        A DALL-E rendering of sparse, deep neural networks in a print style.
    </small>
</div>
<br />

> The _Lottery Ticket Hypothesis_ (LTH) asserts that a randomly initialised overparameterised Deep Neural Network (DNN) contains a sparse subnetwork that, when trained (up to) the same amount as the original network, performs just as well.

The LTH suggests that resetting a DNN to its initialisation helps to find a highly trainable subnetwork.
Neuroevolution Ticket Search (NeTS) is a method for finding a winning ticket for a DNN that uses a genetic algorithm to search for a winning ticket.
It seeks to emulate the workings of Iterative Magnitude Pruning (IMP), introduced alongside the LTH by Frankle and Carbin (2019), by incorporating a fitness signal that includes information both a networks sparsity and trainability.

***This repository conttains an implementation of NeTS written in Python that uses PyTorch models to represent phenotypic networks.***



## Copyright and License

Copyright (c) Alex Jackson, 2023. All rights reserved.

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
