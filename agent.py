import os, sys
import gym
import torch
import cv2
import numpy as np
import random
import time
import schedule
sys.path.append("./")
from network.backbone import BasicNet
from policy_optimizer import Optimizer
from network.dqn import DQN
from utils.dataloader import tnx2batch, batch4net
from utils.atari_wrapper import wrap_rainbow
from utils.trace import PriorTrace
from policy_optimizer import Optimizer
import time

torch.backends.cudnn.benchmark = True

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

class BasicWorker():
    def __init__(self, env_name="PongNoFrameskip-v4", save_interval=1000, max_steps=100000, 
        phase="train", db=None, db_write=None, suffix="default"):
        assert phase in ["train", "valid", "test"], "phase can only be train/test/valid"
        self.phase = phase
        if self.phase == "train":
            self.env = wrap_rainbow(gym.make(env_name), swap=True, phase="train")
        else:
            self.env = wrap_rainbow(gym.make(env_name), swap=True, phase="test")
        self.env_name = env_name
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.db = db
        if db_write:
            assert "db" in db_write.__code__.co_varnames and "data" in db_write.__code__.co_varnames
        self.db_write = db_write
        self.ob = self.reset()
        self.info = {}
        self.cache = []
        self.save_count = 0
        self.video_path = "./video/{}/{}".format(env_name, suffix)
        self.sche = schedule.Sched()
        print("{}\tOb Space: {}\tActions: {}".format(self.env_name, self._shape(), self._na()))

    def reset(self):
        """return ob"""
        return self.env.reset()

    def step(self, a):
        next_ob, rw, done, info = self.env.step(a)
        return next_ob, rw, done, info

    def _simulate_with_train(self):
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
                if len(self.sche):
                    self.sche.wait()
                self.sche.add(None, self.db_write, db=self.db, data=self.cache)
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

    def _simulate_with_test(self, episod=30):
        self.ob = self.reset()
        true_ob = self.env.render(mode="rgb_array")   
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)        
        out = cv2.VideoWriter(os.path.join(self.video_path, "video-{}.avi".format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))),
            fourcc, 25.0, (true_ob.shape[1], true_ob.shape[0]))
        acc_rw, done = 0, False
        count, acc_len = 0, 0
        while count < episod and acc_len < episod*self.max_steps:
            episod_len, done = 0, False
            self.ob = self.reset()
            while not done and episod_len < self.max_steps:
                a = self._action(eps=0.05)
                self.ob, rw, done, _ = self.step(a)
                acc_rw += rw
                episod_len += 1
                acc_len += 1
                true_ob = self.env.render(mode="rgb_array")
                out.write(true_ob)
            count += 1
        out.release()   
        self.ob = self.reset()
        return acc_rw / count

    def __iter__(self):
        return self

    def __next__(self):
        if self.phase == "train":
            return self._simulate_with_train()
        elif self.phase == "valid":
            return self._simulate_with_test(episod=30)
        else:
            return self._simulate_with_test(episod=200)
    
    def _action(self, eps=None):
        return self.env.action_space.sample()

    def _shape(self):
        return self.ob.shape

    def _na(self):
        return self.env.action_space.n
    
    def update(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
        

class DQN_Worker(BasicWorker):
    def __init__(self, env_name="PongNoFrameskip-v4", arch=DQN, backbone=BasicNet, cuda=True,
                save_interval=1000, max_steps=100000, phase="train", db=None, db_write=None,
                suffix="default", **kwargs):
        super(DQN_Worker, self).__init__(env_name, save_interval, max_steps, phase, db, db_write, suffix)
        self.shape = self._shape()
        self.na = self._na()
        self.alg = arch(self.shape, self.na, backbone).eval()
        self.alg.cuda() if cuda == True else None
        self.cuda = cuda
        self.eps = 0

    def _action(self, eps=None):
        eps = self.eps if eps is None else eps
        if random.random() < eps:
            a = self.env.action_space.sample()
        else:
            with torch.no_grad():
                ob = torch.from_numpy(self.ob).cuda().float() if self.cuda else torch.from_numpy(self.ob).float()
                a = self.alg.action(ob).item()
        return a

    def update(self, state_dict=None, eps=None):
        if state_dict is not None:
            self.alg.load_state_dict(state_dict["policy"])
            # if self.write_prior:
            #     self.target.load_state_dict(state_dict["target"])
        if eps is not None:
            self.eps = eps

    def load(self, path):
        self.alg.load_state_dict(torch.load(path))


class PriorDQN_Worker(DQN_Worker):
    """
    todo discount
    """
    def __init__(self, env_name="PongNoFrameskip-v4", arch=DQN, backbone=BasicNet, cuda=True,
                save_interval=1000, max_steps=100000, phase="train", db=None, db_write=None,
                suffix="default", **kwargs):
        super(PriorDQN_Worker, self).__init__(env_name, arch, backbone, cuda, save_interval, max_steps,
                                                phase, db,db_write,suffix)
        self.trace = PriorTrace(discount=0.99) 

    def _action(self, eps=None):
        eps = self.eps if eps is None else eps
        with torch.no_grad():
            ob = torch.from_numpy(self.ob).cuda().float() if self.cuda else torch.from_numpy(self.ob).float()
            v, a = self.alg.value_action(ob)
        if random.random() < eps:
            a = self.env.action_space.sample()
        else:
            a = a.item()
        self.trace.add_value(v)
        self.trace.add_action(a)
        return a

    def _simulate_with_train(self):
        self.ob = self.reset()
        done, episod_len, episod_rw, episod_real_rw = False, 0, 0, 0
        while not done and episod_len < self.max_steps:
            a = self._action()
            next_ob, rw, done, info = self.step(a)
            self.trace.add_reward(rw)
            self.trace.add_done(done)
            self.cache.append({"state": self.ob, "action": a, "next_state": next_ob, "reward": rw, "done": done})
            self.ob = next_ob
            episod_len += 1
            self.save_count += 1
            episod_rw += rw
            episod_real_rw += info["reward"]
            if self.save_count % self.save_interval == 0:
                self._action()
                if len(self.sche):
                    self.sche.wait()
                prior = self.trace.prior()
                assert len(prior) == len(self.cache), "prior len {} not match cache len {}".format(len(prior), len(self.cache))
                self.sche.add(None, self.db_write, db=self.db, data=self.cache, prior=prior)
                self.cache.clear()
                self.save_count = 0
                self.trace.reinit()
        self.info["episod_rw"] = episod_rw
        self.info["episod_real_rw"] = episod_real_rw
        self.info["episod_len"] = episod_len
        if "total_env_steps" in self.info:
            self.info["total_env_steps"] += episod_len
        else:
            self.info["total_env_steps"] = episod_len
        return self.info




    




