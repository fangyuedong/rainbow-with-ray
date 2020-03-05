import os, sys
import lmdb
import shutil
import pickle
import numpy as np
import ray
import time
"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

class ReplayBuffer():
    def push(self, *txns):
        """open replay buffer and push txns"""
        raise NotImplementedError

    def sample(self, num):
        """sample num data"""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError

def _lmdb_read_worker(lmdb_path, idxs):
    with lmdb.open(lmdb_path, map_size=100000000000, readonly=True) as env:
        with env.begin() as txn:
            return [pickle.loads(txn.get(key=str(idx).encode())) for idx in idxs]
        
class LmdbBuffer(ReplayBuffer):
    def __init__(self, lmdb_path, capabilicy=1000000):
        self.lmdb_path = lmdb_path
        self.count = 0
        self.cap = capabilicy
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        os.makedirs(lmdb_path)

    def push(self, data):
        if not isinstance(data, list):
            data = [data]
        with lmdb.open(self.lmdb_path, map_size=100000000000) as env:
            with env.begin(write=True) as txn:
                for x in data:
                    byteformat = pickle.dumps(x)
                    if len(self) == self.cap:
                        txn.delete(key=str(self.count-self.cap).encode())
                    txn.put(key=str(self.count).encode(), value=byteformat)
                    self.count += 1

    def sample(self, num, worker_num=1, shullfer=True, work_func=_lmdb_read_worker):
        print("worker", worker_num)
        if num > len(self):
            print("lmdb:{} don't contain enough data".format(len(self)))
            num = len(self)
        if shullfer == True:
            idxs = np.random.randint(low=max(0,self.count-self.cap), high=self.count, size=num).tolist()
        else:
            idxs = list(range(num))
        if worker_num == 1:
            data = work_func(self.lmdb_path, idxs)
        else:
            list_idxs = [idxs[i:i+num//worker_num] for i in range(0,num,num//worker_num)]
            tsks = [ray.remote(work_func).remote(self.lmdb_path, idxs) for idxs in list_idxs]
            list_data = ray.get(tsks)
            data = []
            [data.extend(x) for x in list_data]

        return data

    def clean(self):
        if os.path.exists(self.lmdb_path):
            shutil.rmtree(self.lmdb_path)

    def __len__(self):
        return self.count - max(0,self.count-self.cap)
