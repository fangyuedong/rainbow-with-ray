import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import LmdbBuffer
from utils.dataloader import Dataloader
from agent import BasicWorker
from policy_optimizer import Optimizer
import torch

class TestCase(unittest.TestCase):
    def test_convergence(self):
        exc_worker = BasicWorker()
        buffer = ray.remote(LmdbBuffer).remote("./ut_lmdb")
        dataloader = Dataloader(buffer, batch_size=256, worker_num=1, batch_num=10, tsk_num=8)
        opt = Optimizer(dataloader, iter_steps=10, update_period=1000)
        for i in range(20):
            traj = exc_worker.__next__()
            ray.get(buffer.push.remote(traj))
            print(i)
        start = time.time()
        for i, loss in opt:
            print(i, loss)
            if i >= 10000:
                break
        print("time: {}".format(time.time()-start))
        buffer.clean.remote()
            

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_convergence"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法