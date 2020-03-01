import os, sys
import lmdb
import shutil
import pickle
import numpy as np
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

class LmdbBuffer(ReplayBuffer):
    def __init__(self, lmdb_path, capabilicy=1000000):
        self.lmdb_path = lmdb_path
        self.count = 0
        self.cap = capabilicy
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
        os.makedirs(lmdb_path)

    def push(self, *txns):
        with lmdb.open(self.lmdb_path, map_size=100000000000) as env:
            with env.begin(write=True) as txn:
                for item in txns:
                    byteformat = pickle.dumps(item)
                    txn.put(key=str(self.count).encode(), value=byteformat)
                    self.count += 1
                # txn.commit() # auto commit when txn close

    def sample(self, num, shullfer=True):
        if num > self.count - max(0,self.count-self.cap):
            print("lmdb:{} don't contain enough data".format(self.count - max(0,self.count-self.cap)))
            num = self.count - max(0,self.count-self.cap)
        if shullfer == True:
            idxs = np.random.randint(low=max(0,self.count-self.cap), high=self.count, size=num).tolist()
        else:
            idxs = list(range(num))
        with lmdb.open(self.lmdb_path, map_size=100000000000, readonly=True) as env:
            with env.begin() as txn:
                data = [pickle.loads(txn.get(key=str(idx).encode())) for idx in idxs]

        return data

    def clean(self):
        if os.path.exists(self.lmdb_path):
            shutil.rmtree(self.lmdb_path)
