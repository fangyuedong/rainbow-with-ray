import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import LmdbBuffer

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
        [worker(self.buffer, [i for _ in range(100)]) for i in range(10)]
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
            return ray.wait([buffer.push.remote(*txns)])
        
        print("\n#Test: Parallel Insert") 
        self.buffer = ray.remote(LmdbBuffer).remote("./ut_lmdb")
        start = time.time()
        task_ids = [ray.remote(worker).remote(self.buffer, [i for _ in range(100)]) for i in range(10)]
        ray.get(task_ids)
        end = time.time()
        print("Time: {}".format(end - start))
        data = self.buffer.sample.remote(1000, shullfer=False)
        data = ray.get(data)
        for i in range(10):
            x = np.array(data[100*i:100*(i+1)])
            assert np.linalg.norm(x - np.ones_like(x) * np.mean(x)) == 0
        ray.wait([self.buffer.clean.remote()])

    def test_SimulatenouslyReadWrite(self):
        def worker(buffer, *txns):
            time.sleep(random.randint(1,10)/500.0)
            return ray.wait([buffer.push.remote(*txns)])
        
        print("\n#Test: Simulatenously Read Write") 
        self.buffer = ray.remote(LmdbBuffer).remote("./ut_lmdb")
        write_tsks = [ray.remote(worker).remote(self.buffer, [i for _ in range(100)]) for i in range(10)]
        ray.wait(write_tsks)
        read_tsks = []
        for _ in range(10):
            time.sleep(0.002)
            read_tsks.append(self.buffer.sample.remote(1000, 4, shullfer=False)) 
        data_list = ray.get(read_tsks)
        for data in data_list:
            print(len(data))
            assert len(data) % 100 == 0
            for i in range(len(data)//100):
                x = np.array(data[100*i:100*(i+1)])
                assert np.linalg.norm(x - np.ones_like(x) * np.mean(x)) == 0
        ray.wait(write_tsks, num_returns=10)
        ray.wait([self.buffer.clean.remote()])

    def test_MultiprocessRead(self):
        def worker(buffer, *txns):
            time.sleep(1)
            return ray.wait([buffer.push.remote(*txns)])
        
        print("\n#Test: Multiprocess Read") 
        self.buffer = ray.remote(LmdbBuffer).remote("./ut_lmdb")
        task_ids = [ray.remote(worker).remote(self.buffer, [i for _ in range(100)]) for i in range(10)]
        ray.get(task_ids)
        data = self.buffer.sample.remote(1000, worker_num=4, shullfer=False)
        data = ray.get(data)
        for i in range(10):
            x = np.array(data[100*i:100*(i+1)])
            assert np.linalg.norm(x - np.ones_like(x) * np.mean(x)) == 0
        ray.wait([self.buffer.clean.remote()])

    def test_TempSerialInsert(self):
        def worker(buffer, *txns):
            time.sleep(1)
            return buffer.push(*txns)
        
        print("\n#Test: Temp Serial Insert")
        self.buffer = LmdbBuffer("./ut_lmdb")
        start = time.time()
        worker(self.buffer, [1,1,1,1,1])
        worker(self.buffer, [2,2,2,2,2])
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
    suite.addTest(TestCase("test_SimulatenouslyReadWrite"))
    suite.addTest(TestCase("test_MultiprocessRead"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法