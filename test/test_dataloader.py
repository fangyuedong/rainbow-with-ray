import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import lmdb_op
from utils.dataloader import Dataloader
from agent import BasicWorker
import torch

class TestCase(unittest.TestCase):

    def test_1_worker(self):
        exc_worker = BasicWorker()
        print(exc_worker._shape())
        buffer = "./ut_lmdb"
        lmdb_op.init(buffer)
        dataloader = Dataloader(buffer, lmdb_op, batch_size=256, worker_num=4, batch_num=10)
        for i in range(20):
            traj, _ = exc_worker.__next__()
            lmdb_op.write(buffer, traj)
            print(i)
        count = 0
        t0 = 0
        for data in dataloader:
            fd = data
            t1 = time.time()
            fd = {k: torch.from_numpy(v) for k, v in fd.items()}
            t2 = time.time()
            fd = {k: v.cuda().float() for k, v in fd.items()}
            t3 = time.time()
            print(t1-t0, t2-t1, t3-t2)
            t0 = t3
            count += 1
            if count == 1000:
                break
        lmdb_op.clean(buffer)
            

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_1_worker"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法