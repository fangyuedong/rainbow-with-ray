import sys,os
import torch
import threading, queue
import time
import ray
import numpy as np
from torch.utils.data._utils.pin_memory import pin_memory
from torch._utils import ExceptionWrapper
sys.path.append("./")
from utils.replay_buffer import DataOp, lmdb_op

MP_STATUS_CHECK_INTERVAL = 5.0
"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""
def tnx2batch(data, batch_size):
    batch_size = min(len(data), batch_size)
    data_split = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    batch_data = [({"state" : np.stack([x["state"] for x in item], axis=0),
        "action": np.stack([x["action"] for x in item], axis=0),
        "next_state": np.stack([x["next_state"] for x in item], axis=0),
        "reward": np.stack([x["reward"] for x in item], axis=0),
        "done": np.stack([x["done"] for x in item], axis=0)}) \
        for item in data_split]   
    return batch_data 

f_cuda = lambda k, x: torch.from_numpy(x).cuda().float() if k != "action" else torch.from_numpy(x).cuda()
f_cpu  = lambda k, x: torch.from_numpy(x).float() if k != "action" else torch.from_numpy(x)

def batch4net(batch, cuda=True):
    f = f_cuda if cuda else f_cpu
    batch = {k: f(k, v) for k, v in batch.items()}
    return batch

def tnx2tensor(tnx):
    return {k: torch.from_numpy(v) for k, v in tnx.items()}

def prior2tensor(prior):
    return  torch.from_numpy(prior)

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
        "done": np.stack([x["done"] for x in item], axis=0)}, item_id, item_idx, np.array(p, dtype=np.float32)) \
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
        self.thread_event = threading.Event()
        self.prefetch_queue = queue.Queue(maxsize=batch_num+1)
        self.pin_queue = queue.Queue(maxsize=batch_num+1)
        self.prefetch_thread = threading.Thread(
            target=self.pre_fetch, 
            args=(self.prefetch_queue, self.thread_event))
        self.pin_thread = threading.Thread(
            target=self.pin_memory, 
            args=(self.prefetch_queue, self.pin_queue, torch.cuda.current_device(), self.thread_event))
        self.prefetch_thread.start()
        self.pin_thread.start()

    def pre_fetch(self, out_queue, done_event):
        while len(self.tsks) < self.worker_num:
            self.tsks.append(pre_fetch_worker.remote(self.data_op.sample, self.dataset, 1, self.batch_size))
        tsk_dones = []
        while not done_event.is_set():
            try:
                if len(tsk_dones) == 1:
                    r = ray.get(tsk_dones[0])[0]
                    tnx, item_id, item_idx, prior = r
                    r = (tnx2tensor(tnx), item_id, item_idx, prior2tensor(prior))
                    out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                    self.tsks.append(pre_fetch_worker.remote(self.data_op.sample, self.dataset, 1, self.batch_size))
            except queue.Full:
                time.sleep(0.001)
                continue
            tsk_dones, self.tsks = ray.wait(self.tsks)
    
    def pin_memory(self, in_queue, out_queue, device_id, done_event):
        torch.set_num_threads(1)
        torch.cuda.set_device(device_id)
        while not done_event.is_set():
            try:
                r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            tnx, item_id, item_idx, prior = r
            data = tnx, prior
            if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
                try:
                    data = pin_memory(data)
                except Exception:
                    data = ExceptionWrapper(
                        where="in pin memory thread for device {}".format(device_id))
                tnx, prior = data
                r = tnx, item_id, item_idx, prior
            while not done_event.is_set():
                try:
                    out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                    break
                except queue.Full:
                    continue
            del r  # save memory

    def __iter__(self):
        return self

    def __next__(self):
        while 1:
            try:
                data = self.pin_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Empty:
                continue
        return data

    def update(self, idxs, priors, check_ids):
        if len(self.update_prior_tsks):
            _, self.update_prior_tsks = ray.wait(self.update_prior_tsks)
        self.update_prior_tsks.append(update_prior_worker.remote(self.data_op.update, self.dataset, idxs, priors, check_ids))

    def __del__(self):
        print("delete")
        self.thread_event.set()
        self.prefetch_thread.join()
        self.pin_thread.join()

