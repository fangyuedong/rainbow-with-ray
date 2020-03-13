import os, sys
import lmdb
import shutil
import pickle
import numpy as np
import time
from collections import namedtuple

"""
define dataset operators
data_op.init(dataset)
data_op.write(dataset, data)
data_op.read(dataset, idxs)
data_op.sample(dataset, nb, shullfer)
data_op.clean(dataset)
data_op.len(dataset)
"""
DataOp = namedtuple("DataOp",["init", "write", "read", "clean", "len", "sample"])

def lmdb_init(lmdb_path, cap=1000000):
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
    os.makedirs(lmdb_path)
    with lmdb.open(lmdb_path, map_size=100000000000) as env:
        with env.begin(write=True) as txn:     
            txn.put(key="num".encode(), value=pickle.dumps(0))
            txn.put(key="head".encode(), value=pickle.dumps(0))
            txn.put(key="cap".encode(), value=pickle.dumps(cap))

def lmdb_write(lmdb_path, data):
    assert os.path.exists(lmdb_path), "lmdb {} not initialized".format(lmdb_path)
    if not isinstance(data, list):
        data = [data]   
    with lmdb.open(lmdb_path, map_size=100000000000) as env:
        with env.begin(write=True) as txn:
            num = pickle.loads(txn.get(key="num".encode()))
            head = pickle.loads(txn.get(key="head".encode()))
            cap = pickle.loads(txn.get(key="cap".encode()))
            for x in data:
                byteformat = pickle.dumps(x)
                txn.put(key=str(head).encode(), value=byteformat)
                head = (head + 1) % cap
                num = min(num + 1, cap)
            txn.put(key="num".encode(), value=pickle.dumps(num))
            txn.put(key="head".encode(), value=pickle.dumps(head))  

def lmdb_read(lmdb_path, idxs):
    assert os.path.exists(lmdb_path), "lmdb {} not initialized".format(lmdb_path)
    with lmdb.open(lmdb_path, map_size=100000000000, readonly=True) as env:
        with env.begin() as txn:
            num = pickle.loads(txn.get(key="num".encode()))
            if isinstance(idxs, list):
                assert max(idxs) < num
                data = [pickle.loads(txn.get(key=str(idx).encode())) for idx in idxs]
            elif isinstance(idxs, tuple):
                assert max(idxs) < num
                data = (pickle.loads(txn.get(key=str(idx).encode())) for idx in idxs)
            elif isinstance(idxs, int):
                assert idxs < num
                data = pickle.loads(txn.get(key=str(idxs).encode()))
            else:
                raise NotImplementedError
    return data

def lmdb_sample(lmdb_path, nb=None, shullfer=True):
    assert os.path.exists(lmdb_path), "lmdb {} not initialized".format(lmdb_path)
    with lmdb.open(lmdb_path, map_size=100000000000, readonly=True) as env:
        with env.begin() as txn:
            num = pickle.loads(txn.get(key="num".encode()))
            if nb == None:
                nb = num
            assert num > 0 and num >= nb, "{} don't have enough data.".format(lmdb_path)
            idxs = np.random.randint(low=0, high=num, size=nb).tolist() if shullfer else list(range(nb))
            data = [pickle.loads(txn.get(key=str(idx).encode())) for idx in idxs]
    return data

def lmdb_clean(lmdb_path):
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)

def lmdb_len(lmdb_path):
    assert os.path.exists(lmdb_path), "lmdb {} not initialized".format(lmdb_path)
    with lmdb.open(lmdb_path, map_size=100000000000, readonly=True) as env:
        with env.begin() as txn:
            return pickle.loads(txn.get(key="num".encode()))

lmdb_op = DataOp(lmdb_init, lmdb_write, lmdb_read, lmdb_clean, lmdb_len, lmdb_sample)