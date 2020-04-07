import os, sys
import gym
import torch
import cv2
import numpy as np
import random
import time
sys.path.append("./")
from network.backbone import BasicNet
from network.dqn import DQN
from utils.atari_wrapper import wrap_rainbow
"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

class BasicWorker():
    def __init__(self, env_name="PongNoFrameskip-v4", save_interval=1000, max_steps=100000, 
        phase="train", db=None, db_write=None):
        self.env = wrap_rainbow(gym.make(env_name), swap=True, phase="train")
        self.env_name = env_name
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.db = db
        self.db_write = db_write
        self.write = lambda x: self.db_write(self.db, x) if db and db_write else None
        self.ob = self.reset()
        self.info = {}
        self.cache = []
        self.save_count = 0
        print("{}\tOb Space: {}\tActions: {}".format(self.env_name, self._shape(), self._na()))

    def reset(self):
        """return ob"""
        return self.env.reset()

    def step(self, a):
        next_ob, rw, done, info = self.env.step(a)
        return next_ob, rw, done, info

    def __iter__(self):
        return self

    def __next__(self):
        self.ob = self.reset()
        done, episod_len, episod_rw, episod_real_rw = False, 0, 0, 0
        while not done and episod_len < self.max_steps:
            a = self._action()
            next_ob, rw, done, info = self.step(a)
            self.cache.append({"state": self.ob, "action": a, "next_state": next_ob, "reward": rw, "done": done})
            self.ob = next_ob
            episod_len += 1
            self.save_count += 1
            episod_rw += rw
            episod_real_rw += info["reward"]
            if self.save_count % self.save_interval == 0:
                self.write(self.cache)
                self.cache.clear()
                self.save_count = 0
        self.info["episod_rw"] = episod_rw
        self.info["episod_real_rw"] = episod_real_rw
        self.info["episod_len"] = episod_len
        if "total_env_steps" in self.info:
            self.info["total_env_steps"] += episod_len
        else:
            self.info["total_env_steps"] = episod_len
        return self.info
     
    # def traj(self):
    #     return self.fetch
    
    def _action(self, eps=None):
        return self.env.action_space.sample()

    def _shape(self):
        return self.ob.shape

    def _na(self):
        return self.env.action_space.n
    
    def update(self):
        raise NotImplementedError

    def save(self, video_path):
        self.ob = self.reset()
        true_ob = self.env.render(mode="rgb_array")
        episod_len = 0     
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists(video_path):
            os.makedirs(video_path)        
        out = cv2.VideoWriter(os.path.join(video_path, "video-{}.avi".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))),
            fourcc, 25.0, (true_ob.shape[1], true_ob.shape[0]))
        acc_rw, real_acc_rw, done = 0, 0, False
        while not done and episod_len < self.max_steps:
            a = self._action(eps=.0)
            self.ob, rw, done, info = self.step(a)
            acc_rw += rw
            episod_len += 1
            real_acc_rw += info["reward"]
            true_ob = self.env.render(mode="rgb_array")
            out.write(true_ob)
        out.release()   
        self.ob = self.reset()
        self.info["episod_rw"] = acc_rw
        self.info["episod_real_rw"] = real_acc_rw
        self.info["episod_len"] = episod_len
        return self.info
        

class DQN_Worker(BasicWorker):
    def __init__(self, env_name="PongNoFrameskip-v4", arch=DQN, backbone=BasicNet, cuda=True,
                save_interval=1000, max_steps=100000, phase="train", db=None, db_write=None):
        super(DQN_Worker, self).__init__(env_name, save_interval, max_steps, phase, db, db_write)
        self.shape = self._shape()
        self.na = self._na()
        self.alg = arch(self.shape, self.na, backbone).eval()
        self.alg.cuda() if cuda == True else None
        self.cuda = cuda
        self.eps = 0

    def _action(self, eps=None):
        eps = self.eps if eps is None else eps
        with torch.no_grad():
            ob = torch.from_numpy(self.ob).cuda().float() if self.cuda else torch.from_numpy(self.ob).float()
            net_a = self.alg.action(ob).item()
        rand_a = self.env.action_space.sample()
        a = rand_a if random.random() < eps else net_a
        return a


    def update(self, state_dict=None, eps=None):
        if state_dict is not None:
            self.alg.load_state_dict(state_dict)
        if eps is not None:
            self.eps = eps
        





    




