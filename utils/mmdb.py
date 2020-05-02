import ray
import numpy as np
import gzip
import pickle as pkl

def default_func(db, nb=None, shullfer=True):
    num = len(db)
    if nb == None:
        nb = num
    assert num > 0 and num >= nb, "nb {} db {} don't have enough data.".format(nb, num)
    idxs = np.random.randint(low=0, high=num, size=nb).tolist() if shullfer else list(range(nb))
    return idxs

class Mmdb():
    def __init__(self, cap=1000000, memory_limit=3*1024**3, mm_count=None, sample_func=default_func):
        self.cap = cap
        self.mm_limit = memory_limit
        self.tail = 0
        self.count = 0
        self.mm_cost = 0
        self.db = dict()
        self.data_id = dict()
        self.mm_count = lambda x: 0 if mm_count == None else mm_count(x)
        self.sample_func = sample_func

    def write(self, data):
        if not isinstance(data, (list, tuple)):
            data = [data]  
        for item in data:
            if self.tail in self.db:
                self.mm_cost += self.mm_count(item) - self.mm_count(self.db[self.tail])
            else:
                self.mm_cost += self.mm_count(item)
            assert self.mm_cost <= self.mm_limit, "db mm_cost {}byte out of mm_limit {}byte".\
                format(self.mm_cost, self.mm_limit)
            self.db[self.tail] = item
            self.data_id[self.tail] = self.count
            self.count += 1
            self.tail += 1
            self.tail %= self.cap

    def read(self, idxs):
        if isinstance(idxs, list):
            assert max(idxs) < len(self), "idx {} out of db len {}".format(max(idxs), len(self))
            data = [self.db[idx] for idx in idxs]
            data_id = [self.data_id[idx] for idx in idxs]
        elif isinstance(idxs, tuple):
            assert max(idxs) < len(self), "idx {} out of db len {}".format(max(idxs), len(self))
            data = (self.db[idx] for idx in idxs)
            data_id = (self.data_id[idx] for idx in idxs)
        elif isinstance(idxs, int):
            assert idxs < len(self), "idx {} out of db len {}".format(idxs, len(self))
            data = self.db[idxs]
            data_id = self.data_id[idxs]
        else:
            raise NotImplementedError
        return data, data_id, idxs

    def sample(self, nb, shullfer=True):
        idxs = self.sample_func(self, nb, shullfer)
        return self.read(idxs)

    def __len__(self):
        return len(self.db)

    def check_id(self, idxs, data_ids):
        return [self.data_id[idx] == data_id for idx, data_id in zip(idxs, data_ids)]

def mmdb_init(path, cap=1000000):
    return ray.remote(Mmdb).remote(cap, mm_count=len)

def mmdb_write(db, data, **kwargs):
    if not isinstance(data, (list, tuple)):
        data = [data]
    zip_data = [gzip.compress(pkl.dumps(x), 7) for x in data]
    assert isinstance(db, ray.actor.ActorHandle) and \
        db._ray_actor_creation_function_descriptor.class_name == "Mmdb"
    tsk = db.write.remote(zip_data)
    ray.wait([tsk])

def mmdb_read(db, idxs):
    assert isinstance(db, ray.actor.ActorHandle) and \
        db._ray_actor_creation_function_descriptor.class_name == "Mmdb"
    zip_data, data_id, idx = ray.get(db.read.remote(idxs))
    data = [pkl.loads(gzip.decompress(x)) for x in zip_data]
    return data, data_id, idx

def mmdb_sample(db, nb=None, shullfer=True):
    assert isinstance(db, ray.actor.ActorHandle) and \
        db._ray_actor_creation_function_descriptor.class_name == "Mmdb"
    zip_data, data_id, idx = ray.get(db.sample.remote(nb, shullfer))
    data = [pkl.loads(gzip.decompress(x)) for x in zip_data]
    return data, data_id, idx
    
def mmdb_update(db, **kwargs):
    pass

def mmdb_clean(db):
    pass

def mmdb_len(db):
    return ray.get(db.__len__.remote())

        