import sys,os
import numpy as np
import torch
import ray
import time
sys.path.append("./")
from utils.replay_buffer import DataOp, lmdb_op

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

@ray.remote
def pre_fetch_worker(work_func, dataset, num, batch_size):
    data, data_id, idx, p = work_func(dataset, num*batch_size)
    data_split = [data[i:i+batch_size] for i in range(0, num*batch_size, batch_size)]
    data_id_split = [data_id[i:i+batch_size] for i in range(0, num*batch_size, batch_size)]
    idx_split = [idx[i:i+batch_size] for i in range(0, num*batch_size, batch_size)]
    p_split = [p[i:i+batch_size] for i in range(0, num*batch_size, batch_size)]
    data = [({"state" : np.stack([x["state"] for x in item], axis=0),
        "action": np.stack([x["action"] for x in item], axis=0),
        "next_state": np.stack([x["next_state"] for x in item], axis=0),
        "reward": np.stack([x["reward"] for x in item], axis=0),
        "done": np.stack([x["done"] for x in item], axis=0)}, item_id, item_idx, np.array(p)) \
        for item, item_id, item_idx, p in zip(data_split, data_id_split, idx_split, p_split)]
    return data
        
@ray.remote
def update_prior_worker(work_func, dataset, idxs, priors, check_ids):
    work_func(dataset, idxs, priors, check_ids)

class Dataloader():
    def __init__(self, dataset, data_op, worker_num=1, batch_size=64, batch_num=10):
        self.dataset = dataset
        assert isinstance(data_op, DataOp)
        self.data_op = data_op
        self.worker_num = worker_num
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.tsks = []
        self.cache = []
        self.update_prior_tsks = []

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.tsks) < self.worker_num:
            self.tsks.append(pre_fetch_worker.remote(self.data_op.sample, self.dataset, self.batch_num, self.batch_size))

        if len(self.cache) == 0:
            tsk_dones, self.tsks = ray.wait(self.tsks)
            assert len(tsk_dones) == 1
            self.cache = ray.get(tsk_dones[0])
            return self.cache.pop(0)
        else:
            return self.cache.pop(0)  

    def update(self, idxs, priors, check_ids):
        if len(self.update_prior_tsks):
            _, self.update_prior_tsks = ray.wait(self.update_prior_tsks)
        self.update_prior_tsks.append(update_prior_worker.remote(self.data_op.update, self.dataset, idxs, priors, check_ids))

