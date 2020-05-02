import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import pmdb_op as db_op
from utils.dataloader import Dataloader
from agent import BasicWorker
import torch

class TestCase(unittest.TestCase):

    def test_1_worker(self):
        buffer = "./ut_lmdb"
        buffer = db_op.init(buffer)
        exc_worker = BasicWorker(db=buffer, db_write=db_op.write)
        dataloader = Dataloader(buffer, db_op, batch_size=256, worker_num=4, batch_num=10)
        for _ in range(20):
            _ = exc_worker.__next__()
            print(db_op.len(buffer))
        count = 0
        t0 = time.time()
        for data, _, _ in dataloader:
            fd = {k: torch.from_numpy(v) for k, v in data.items()}
            fd = {k: v.cuda().float() for k, v in fd.items()}
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