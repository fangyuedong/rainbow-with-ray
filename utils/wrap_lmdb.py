import os,sys
import shutil
import lmdb
import pickle

def lmdb_init(path, cap=1000000):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    with lmdb.open(path, map_size=100000000000) as env:
        with env.begin(write=True) as txn:     
            txn.put(key="num".encode(), value=pickle.dumps(0))
            txn.put(key="head".encode(), value=pickle.dumps(0))
            txn.put(key="cap".encode(), value=pickle.dumps(cap))
    return path

def lmdb_write(db, data):
    assert os.path.exists(db), "lmdb {} not initialized".format(db)
    if not isinstance(data, list):
        data = [data]   
    with lmdb.open(db, map_size=100000000000) as env:
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

def lmdb_read(db, idxs):
    assert os.path.exists(db), "lmdb {} not initialized".format(db)
    with lmdb.open(db, map_size=100000000000, readonly=True) as env:
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

def lmdb_sample(db, nb=None, shullfer=True):
    assert os.path.exists(db), "lmdb {} not initialized".format(db)
    with lmdb.open(db, map_size=100000000000, readonly=True) as env:
        with env.begin() as txn:
            num = pickle.loads(txn.get(key="num".encode()))
            if nb == None:
                nb = num
            assert num > 0 and num >= nb, "{} don't have enough data.".format(db)
            idxs = np.random.randint(low=0, high=num, size=nb).tolist() if shullfer else list(range(nb))
            data = [pickle.loads(txn.get(key=str(idx).encode())) for idx in idxs]
    return data

def lmdb_clean(db):
    if os.path.exists(db):
        shutil.rmtree(db)

def lmdb_len(db):
    assert os.path.exists(db), "lmdb {} not initialized".format(db)
    with lmdb.open(db, map_size=100000000000, readonly=True) as env:
        with env.begin() as txn:
            return pickle.loads(txn.get(key="num".encode()))