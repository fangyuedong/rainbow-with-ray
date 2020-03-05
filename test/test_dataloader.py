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

class TestCase(unittest.TestCase):
    #在setUp()方法中进行测试前的初始化工作。
    def setUp(self):   
        pass

    #并在tearDown()方法中执行测试后的清除工作，setUp()和tearDown()都是TestCase类中定义的方法。
    def tearDown(self):
        pass
 
    def test_1_worker(self):
        exc_worker = BasicWorker()
        buffer = ray.remote(LmdbBuffer).remote("./ut_lmdb")
        dataloader = Dataloader(buffer, batch_size=256, worker_num=1)
        for i in range(20):
            traj = exc_worker.exc()
            ray.get(buffer.push.remote(traj))
            print(i)
        count = 0
        start = time.time()
        for _ in dataloader:
            end = time.time()
            print(end - start)
            start = end
            count += 1
            time.sleep(0.2)
            if count == 20:
                break
        buffer.clean.remote()
            

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_1_worker"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法