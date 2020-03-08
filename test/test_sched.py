import sys
import unittest 
import ray
import time
sys.path.append("./")
from agent import DQN_Worker
from schedule import Sched

class TestCase(unittest.TestCase):

    def test_sched_1_actor(self):
        self.sche = Sched()
        self.worker0 = ray.remote(DQN_Worker).options(num_gpus=0.1).remote()
        for _ in range(5):
            self.sche.add(self.worker0, method="__next__")
            tsk_ids, infos = self.sche.wait(1)
            tsk_id, info = tsk_ids[0], infos[0]
            print("{}: {}.{}({})".format(tsk_id,info.class_name, info.method, info.params))


def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_sched_1_actor"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法