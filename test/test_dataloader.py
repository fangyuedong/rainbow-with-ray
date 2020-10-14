import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import mmdb_op as db_op
from utils.dataloader import Dataloader
from agent import BasicWorker, DQN_Worker
import torch

class TestCase(unittest.TestCase):

    def test_1_worker(self):
        buffer = "./ut_lmdb"
        buffer = db_op.init(buffer)
        # exc_worker = BasicWorker(db=buffer, db_write=db_op.write)
        exc_worker = DQN_Worker(db=buffer, db_write=db_op.write)
        # exc_worker.update(eps=0.9)
        dataloader = Dataloader(buffer, db_op, batch_size=256, worker_num=8, batch_num=20)
        # for _ in range(20):
        t0 = time.time()
        while db_op.len(buffer) < 10000:
            _ = exc_worker.__next__()
            # print(db_op.len(buffer))
        print(time.time() - t0)
        count = 0
        t0 = time.time()
        for data, _, _, _ in dataloader:
            fd = {k: torch.from_numpy(v) for k, v in data.items()}
            fd = {k: v.cuda().float() for k, v in fd.items()}
            time.sleep(0.02)
            count += 1
            if count == 1000:
                break
        print(time.time() - t0)
        db_op.clean(buffer)
            

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_1_worker"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法