import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import lmdb_op

class TestCase(unittest.TestCase):
    def test_SerialInsert(self):
        print("\n#Test: Serial Insert")
        lmdb_op.init("./ut_lmdb")
        start = time.time()
        [lmdb_op.write("./ut_lmdb", [i for _ in range(100)]) for i in range(10)]
        end = time.time()
        print("Time: {}".format(end - start))
        data = lmdb_op.sample("./ut_lmdb", 1000, shullfer=False)
        target = []
        for i in range(10):
            target += [i for _ in range(100)]
        assert data == target, "data{} not equal target{}".format(data, target)
        lmdb_op.clean("./ut_lmdb")

    def test_ParallelInsert(self):        
        print("\n#Test: Parallel Insert") 
        lmdb_op.init("./ut_lmdb")
        start = time.time()
        task_ids = [ray.remote(lmdb_op.write).remote("./ut_lmdb", [i for _ in range(100)]) for i in range(10)]
        ray.get(task_ids)
        end = time.time()
        print("Time: {}".format(end - start))
        data = lmdb_op.sample("./ut_lmdb", 1000, shullfer=False)
        time.sleep(1)
        for i in range(10):
            x = np.array(data[100*i:100*(i+1)])
            assert np.linalg.norm(x - np.ones_like(x) * np.mean(x)) == 0
        lmdb_op.clean("./ut_lmdb")

    def test_SimulatenouslyReadWrite(self):
        def worker(txns):
            time.sleep(random.randint(1,10)/500.0)
            lmdb_op.write("./ut_lmdb", txns)
            return
        
        print("\n#Test: Simulatenously Read Write") 
        lmdb_op.init("./ut_lmdb")
        write_tsks = [ray.remote(worker).remote([i for _ in range(100)]) for i in range(10)]
        ray.wait(write_tsks)
        read_tsks = []
        for _ in range(10):
            time.sleep(0.002)
            read_tsks.append(ray.remote(lmdb_op.sample).remote("./ut_lmdb", shullfer=False)) 
        data_list = ray.get(read_tsks)
        time.sleep(1)
        for data in data_list:
            print(len(data))
            assert len(data) % 100 == 0
            for i in range(len(data)//100):
                x = np.array(data[100*i:100*(i+1)])
                assert np.linalg.norm(x - np.ones_like(x) * np.mean(x)) == 0
        ray.wait(write_tsks, num_returns=10)
        lmdb_op.clean("./ut_lmdb")

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_SerialInsert"))
    suite.addTest(TestCase("test_ParallelInsert"))
    suite.addTest(TestCase("test_SimulatenouslyReadWrite"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法