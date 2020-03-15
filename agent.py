from network.backbone import BasicNet
from network.dqn import DQN
from atari_wrapper import wrap_rainbow
import os
import gym
import torch
import cv2
import numpy as np
import random
import time

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

class BasicWorker():
    def __init__(self, env_name="PongNoFrameskip-v4", output_interval=1000, max_steps=100000, phase="train"):
        self.env = wrap_rainbow(gym.make("PongNoFrameskip-v4"), swap=True, phase="train")
        self.env_name = "PongNoFrameskip-v4"
        self.output_interval = output_interval
        self.max_steps = max_steps
        self.ob = self.reset()
        print("{}\tOb Space: {}\tActions: {}".format(self.env_name, self._shape(), self._na()))
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

    def save(self, video_path):
        self.ob = self.reset()
        self.episod_len = 0     
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists(video_path):
            os.makedirs(video_path)        
        out = cv2.VideoWriter(os.path.join(video_path, "video-{}.avi".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))),
            fourcc, 25.0, (160,210))
        acc_rw, done = 0, False
        while not done and self.episod_len < self.max_steps:
            a = self._action()
            self.ob, rw, done = self.step(a)
            acc_rw += rw
            true_ob = self.env.render(mode="rgb_array")
            out.write(true_ob)
        out.release()   
        self.ob = self.reset()
        self.episod_len = 0
        self.episod_rw = 0  


class DQN_Worker(BasicWorker):
    def __init__(self, env_name="PongNoFrameskip-v4", arch=DQN, backbone=BasicNet, cuda=True,
                output_interval=1000, max_steps=100000, phase="train"):
        super(DQN_Worker, self).__init__(env_name, output_interval, max_steps, phase)
        self.shape = self._shape()
        self.na = self._na()
        self.alg = arch(self.shape, self.na, backbone).eval()
        self.alg.cuda() if cuda == True else None
        self.cuda = cuda
        self.eps = 0

    def _action(self):
        with torch.no_grad():
            ob = torch.from_numpy(self.ob).cuda().float() if self.cuda else torch.from_numpy(self.ob).float()
            net_a = self.alg.action(ob).item()
        rand_a = self.env.action_space.sample()
        a = rand_a if random.random() < self.eps else net_a
        return a


    def update(self, state_dict=None, eps=None):
        if state_dict is not None:
            self.alg.load_state_dict(state_dict)
        if eps is not None:
            self.eps = eps
        





    




