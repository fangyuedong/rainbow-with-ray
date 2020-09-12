# Distributed reinforcement learning based on Ray

A PyTorch implementation of reinforcement lerning algorithms

## What implemented
* DQN
* DDQN
* Distributed
* Prior replaybuffer

## What implementing
* R2D2

## What to implement
* Multi-step learning
* ...

## What special compared to other implementions
* Compressing is used before observations saved into replay buffer, so only 1~3GB RAM will be used to keep 1M history observations.

## Some result
* DQN

* DDQN

    | Game            | Zaxxon                          | Asterix                           | WizardOfWor                               |
    |:---------------:|:-------------------------------:|:---------------------------------:|:-----------------------------------------:|
    |Result           |![avatar](./exp/ddqn/Zaxxon.gif) |![avatar](./exp/ddqn/Asterix.gif)  |![avatar](./exp/ddqn/WizardOfWor.gif)      |
    |Noop Start Score | [11610](./exp/ddqn/Zaxxon.png)  | [27287](./exp/ddqn/Asterix.png)   | [6627](./exp/ddqn/WizardOfWor.png)        |

* Prior DDQN

    | Game            | Asterix                                 |
    |:---------------:|:---------------------------------------:|
    |Result           |![avatar](./exp/prior_ddqn/Asterix.gif)  |
    |Noop Start Score | [39288](./exp/prior_ddqn/Asterix.png)   |

## How to use

* DQN with an uniform replay buffer

    nohup python -u main.py Asterix --alg=DQN --buffer=mmdb --num_agents=4 --num_loaders=6 --batch_size=256 --lr=0.625e-4 --suffix="DQN" --speed=8 >train.txt 2>&1 &

* DDQN with an uniform replay buffer

    nohup python -u main.py Asterix --alg=DDQN --buffer=mmdb --num_agents=4 --num_loaders=6 --batch_size=256 --lr=0.625e-4 --suffix="DDQN" --speed=8 >train.txt 2>&1 &

* DDQN with a prior replay buffer

    nohup python -u main.py Asterix --alg=DDQN --buffer=pmdb --num_agents=4 --num_loaders=6 --batch_size=256 --lr=0.150e-4 --suffix="DDQN" --speed=8 >train.txt 2>&1 &

* Test with a trained model

    python main.py WizardOfWor --test --suffix="DDQN_gn_normal0" --resume ./model/DQN_BasicNet/WizardOfWorNoFrameskip-v4/DDQN_gn_normal0/iter_3600000K.pkl


