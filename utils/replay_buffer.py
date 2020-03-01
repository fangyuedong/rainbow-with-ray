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


# def lmdb_read_worker(env, idxs):
#     with env.begin() as txn:
#         return [pickle.loads(txn.get(key=str(idx).encode())) for idx in idxs]

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
        assert self.count - max(0,self.count-self.cap) >= num, "lmdb:{} don't contain enough data".format(self.count - max(0,self.count-self.cap))
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


import unittest 
import time
import ray

class TestCase(unittest.TestCase):
    #在setUp()方法中进行测试前的初始化工作。
    def setUp(self):   
        pass

    #并在tearDown()方法中执行测试后的清除工作，setUp()和tearDown()都是TestCase类中定义的方法。
    def tearDown(self):
        pass
 
    def test_SerialInsert(self):
        def worker(buffer, *txns):
            time.sleep(1)
            return buffer.push(*txns)
        
        print("\n#Test: Serial Insert")
        self.buffer = LmdbBuffer("./ut_lmdb")
        start = time.time()
        [worker(self.buffer, *[i for _ in range(100)]) for i in range(10)]
        end = time.time()
        print("Time: {}".format(end - start))
        data = self.buffer.sample(1000, shullfer=False)
        target = []
        for i in range(10):
            target += [i for _ in range(100)]
        assert data == target, "data{} not equal target{}".format(data, target)
        self.buffer.clean()

    def test_ParallelInsert(self):
        def worker(buffer, *txns):
            time.sleep(1)
            return buffer.push.remote(*txns)
        
        print("\n#Test: Parallel Insert") 
        self.buffer = ray.remote(LmdbBuffer).remote("./ut_lmdb")
        start = time.time()
        task_ids = [ray.remote(worker).remote(self.buffer, *[i for _ in range(100)]) for i in range(10)]
        ray.get(task_ids)
        end = time.time()
        print("Time: {}".format(end - start))
        data = self.buffer.sample.remote(1000, shullfer=False)
        data = ray.get(data)
        for i in range(10):
            x = np.array(data[100*i:100*(i+1)])
            assert np.linalg.norm(x - np.ones_like(x) * np.mean(x)) == 0
        self.buffer.clean.remote()
        
    def test_TempSerialInsert(self):
        def worker(buffer, *txns):
            time.sleep(1)
            return buffer.push(*txns)
        
        print("\n#Test: Temp Serial Insert")
        self.buffer = LmdbBuffer("./ut_lmdb")
        start = time.time()
        worker(self.buffer, *[1,1,1,1,1])
        worker(self.buffer, *[2,2,2,2,2])
        end = time.time()
        print("Time: {}".format(end - start))
        self.buffer.clean()

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_TempSerialInsert"))
    suite.addTest(TestCase("test_SerialInsert"))
    suite.addTest(TestCase("test_ParallelInsert"))
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法