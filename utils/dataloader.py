import sys,os
import numpy as np
import torch
import ray
import time
sys.path.append("./")
from utils.replay_buffer import LmdbBuffer

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

@ray.remote
def pre_fetch_worker(dataset_actor, num, batch_size, worker_num=1):
    tsk_id = dataset_actor.sample.remote(num*batch_size, worker_num=worker_num)
    data = ray.get(tsk_id)
    data_split = [data[i:i+batch_size] for i in range(0, num*batch_size, batch_size)]
    data = [{"state" : np.stack([x["state"] for x in item], axis=0),
        "action": np.stack([x["action"] for x in item], axis=0),
        "next_state": np.stack([x["next_state"] for x in item], axis=0),
        "reward": np.stack([x["reward"] for x in item], axis=0),
        "done": np.stack([x["done"] for x in item], axis=0)} for item in data_split]

    return data

class Dataloader():
    def __init__(self, dataset, worker_num=1, batch_size=64, batch_num=10, tsk_num=2):
        self.dataset = dataset
        self.worker_num = worker_num
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.tsk_num = tsk_num
        self.tsks = []
        self.cache = []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while len(self.tsks) < self.tsk_num:
            self.tsks.append(pre_fetch_worker.remote(self.dataset, self.batch_num, self.batch_size, self.worker_num))

        if len(self.cache) == 0:
            tsk_dones, self.tsks = ray.wait(self.tsks)
            assert len(tsk_dones) == 1
            self.cache = ray.get(tsk_dones[0])
            data = self.cache[0]
            self.cache.pop(0)
            return data
        else:
            data = self.cache[0]
            self.cache.pop(0)
            return data
        
        


