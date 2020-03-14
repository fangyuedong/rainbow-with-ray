from network.backbone import BasicNet
from network.dqn import DQN
from atari_wrapper import wrap_rainbow
import gym
import torch
import cv2
import numpy as np
import random

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

class BasicWorker():
    def __init__(self, env_name="PongNoFrameskip-v4", workid="work_{:0>2d}".format(0), output_interval=1000, max_steps=100000, phase="train", debug=False):
        self.env = wrap_rainbow(gym.make("PongNoFrameskip-v4"), swap=True, phase="train")
        self.workid = workid
        self.env_name = "PongNoFrameskip-v4"
        self.output_interval = output_interval
        self.debug = debug
        self.max_steps = max_steps
        print("{}\t{}\tActions: {}".format(self.workid, self.env_name, self._na))
        self.ob = self.reset()
        self.episod_rw = 0
        self.episod_len = 0

    def reset(self):
        """return ob"""
        return self.env.reset()

    def step(self, a):
        next_ob, rw, done, _ = self.env.step(a)
        return next_ob, rw, done

    def __iter__(self):
        return self

    def __next__(self):
        done, count, cache = False, 0, []
        while not done and count < self.output_interval and self.episod_len < self.max_steps:
            a = self._action()
            next_ob, rw, done = self.step(a)
            if self.debug:
                cv2.imshow("video", self.ob[0,:,:])
                cv2.waitKey(25)
            cache.append({"state": self.ob, "action": a, "next_state": next_ob, "reward": rw, "done": done})
            self.ob = next_ob
            self.episod_len += 1
            self.episod_rw += rw
            count += 1
        if done or self.episod_len >= self.max_steps:
            self.ob = self.reset()
            self.episod_len = 0
            sum_rw = self.episod_rw
            self.episod_rw = 0
            return cache, sum_rw
        else:
            return cache, None
    
    def _action(self):
        return self.env.action_space.sample()

    def _shape(self):
        return self.ob.shape

    def _na(self):
        return self.env.action_space.n
    
    def update(self):
        raise NotImplementedError


class DQN_Worker(BasicWorker):
    def __init__(self, env_name="PongNoFrameskip-v4", workid="work_{:0>2d}".format(0), 
                arch=DQN, backbone=BasicNet, cuda=True,
                output_interval=1000, max_steps=100000, phase="train", debug=False):
        super(DQN_Worker, self).__init__(env_name, workid, output_interval, max_steps, phase, debug)
        self.shape = self._shape()
        self.na = self._na()
        self.alg = arch(self.shape, self.na, backbone).eval()
        self.alg.cuda() if cuda == True else None
        self.cuda = cuda

    def _action(self):
        with torch.no_grad():
            ob = torch.from_numpy(self.ob).cuda().float() if self.cuda else torch.from_numpy(self.ob).float()
            return self.alg.action(ob).item()

    def update(self, state_dict):
        self.alg.load_state_dict(state_dict)
        





    




