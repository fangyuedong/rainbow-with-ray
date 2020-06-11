# Distributed reinforcement learning based on Ray

A PyTorch implementation of reinforcement lerning algorithms

## What implemented
* DQN
* DDQN
* Distributed

## What implementing
* Prior replaybuffer

## What to implement
* Multi-step learning
* R2D2
* ...

## What special compared to other implementions
* Compressing is used before observations saved into replay buffer, so only 1~3GB RAM will be used to keep 1M history observations.

## Some Result
* DQN

* DDQN

    [Zaxxon](./exp/ddqn/Zaxxon.png)

![avatar](./exp/ddqn/Zaxxon.gif)