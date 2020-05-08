import os, sys
import math
import random
import gzip
import pickle as pkl
import ray
sys.path.append("./")
from utils.mmdb import Mmdb

class SegTree():
    def __init__(self, max_num=1000**2, alpha=0.6, maxp=None):
        self.max_num = max_num
        self.alpha = alpha
        self.index = 0
        self.history_len = 0
        self.struct_len = 2**math.ceil(math.log(max_num, 2))
        self.sum_tree = [0.0 for _ in range(2*self.struct_len-1)]
        self.maxp = maxp**alpha

    def _propagate(self, idx, value):
        parent = idx
        self.sum_tree[parent] = value**self.alpha if self.maxp==None else min(value**self.alpha, self.maxp)
        while parent > 0:
            parent = (parent-1)//2
            left = 2*parent+1
            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[left+1]
        
    def update(self, idxs, values):
        for idx, value in zip(idxs, values):self._propagate(idx+self.struct_len-1, value)

    def append(self, values):
        for value in values:
            self._propagate(self.index+self.struct_len-1, value)
            self.index = (self.index+1)%self.max_num
        self.history_len += len(values)

    def _retrieve(self, value):
        assert value < self.sum_tree[0]
        top = 0
        left = 2*top+1
        l = len(self.sum_tree)
        while left < l:
            left_value = self.sum_tree[left]
            if value < left_value:
                top = left
            else:
                value -= left_value
                top = left+1
            left = 2*top+1
        return top

    def find(self, values):
        return [self._retrieve(value)-self.struct_len+1 for value in values]

    def __getitem__(self, idx):
        return self.sum_tree[idx+self.struct_len-1]
    
    def __len__(self):
        return min(self.history_len, self.max_num)

    def total(self):
        return self.sum_tree[0]


def default_func(db, nb):
    assert isinstance(db, Pmdb)
    total = db.tree.total()
    values = [(random.random() + i) * total / nb for i in range(nb)]
    idxs = db.tree.find(values)
    return idxs

class Pmdb():
    def __init__(self, cap=1000000, memory_limit=3*1024**3, mm_count=None, sample_func=default_func, maxp=None):
        self.mmdb = Mmdb(cap, memory_limit, mm_count)
        self.tree = SegTree(cap, maxp=maxp)
        self.sample_func = sample_func
        self.maxp = maxp
        assert len(self.mmdb) == len(self.tree)

    def write(self, data, prior=None):
        assert not prior or not self.maxp
        if not prior and self.maxp:
            prior = [self.maxp for _ in range(len(data))]
        self.mmdb.write(data)
        self.tree.append(prior)
        assert len(self.mmdb) == len(self.tree)

    def read(self, idxs):
        return self.mmdb.read(idxs)

    def sample(self, nb):
        idxs = self.sample_func(self, nb)
        total = self.tree.total()
        ps = [self.tree[idx]/total for idx in idxs]
        return self.read(idxs) + (ps,)

    def update(self, idxs, priors, check_ids):
        checks = self.mmdb.check_id(idxs, check_ids)
        update_ids = [idx for idx, check in zip(idxs, checks) if check]
        update_priors = [prior for prior, check in zip(priors, checks) if check]
        self.tree.update(update_ids, update_priors)

    def __len__(self):
        return len(self.mmdb)

    def config(self):
        config = self.mmdb.config()
        return config

def pmdb_init(path, cap=1000000):
    return ray.remote(Pmdb).remote(cap, mm_count=len, maxp=1.0)

def pmdb_write(db, data, prior=None):
    if not isinstance(data, (list, tuple)):
        data = [data]
    zip_data = [gzip.compress(pkl.dumps(x), 7) for x in data]
    assert isinstance(db, ray.actor.ActorHandle) and \
        db._ray_actor_creation_function_descriptor.class_name == "Pmdb"
    ray.get(db.write.remote(zip_data, prior))

def pmdb_read(db, idxs):
    assert isinstance(db, ray.actor.ActorHandle) and \
        db._ray_actor_creation_function_descriptor.class_name == "Pmdb"
    zip_data, data_id, idx = ray.get(db.read.remote(idxs))
    data = [pkl.loads(gzip.decompress(x)) for x in zip_data]
    return data, data_id, idx

def pmdb_sample(db, nb, **kwargs):
    assert isinstance(db, ray.actor.ActorHandle) and \
        db._ray_actor_creation_function_descriptor.class_name == "Pmdb"
    zip_data, data_id, idx, p = ray.get(db.sample.remote(nb))
    data = [pkl.loads(gzip.decompress(x)) for x in zip_data]
    return data, data_id, idx, p
    
def pmdb_update(db, idxs, priors, check_ids):
    assert isinstance(db, ray.actor.ActorHandle) and \
        db._ray_actor_creation_function_descriptor.class_name == "Pmdb"
    ray.get(db.update.remote(idxs, priors, check_ids))   

def pmdb_clean(db):
    pass

def pmdb_len(db):
    return ray.get(db.__len__.remote())

def pmdb_config(db):
    return ray.get(db.config.remote())
        



    

        