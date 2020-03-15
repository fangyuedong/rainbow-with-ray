import sys
import unittest 
import ray
import time
sys.path.append("./")
from agent import DQN_Worker
from schedule import Sched
from utils.replay_buffer import lmdb_op
from utils.dataloader import Dataloader
from policy_optimizer import Optimizer
import torch

class TestCase(unittest.TestCase):

    def test_sched_1_actor(self):
        self.sche = Sched()
        self.worker0 = ray.remote(DQN_Worker).options(num_gpus=0.1).remote()
        for _ in range(5):
            self.sche.add(self.worker0, method="__next__")
            tsk_ids, infos = self.sche.wait(1)
            tsk_id, info = tsk_ids[0], infos[0]
            print("{}: {}.{}({})".format(tsk_id,info.class_name, info.method, info.params))

    def test_sche_opt(self):
        self.sche = Sched()
        buffer = "./ut_lmdb"
        print(lmdb_op.len(buffer))
        dataloader = Dataloader(buffer, lmdb_op, worker_num=3, batch_size=64, batch_num=40)
        opt = ray.remote(Optimizer).options(num_gpus=0.3).remote(dataloader, iter_steps=10, update_period=10000)
        t0 = time.time()
        self.sche.add(opt, "__call__")
        self.sche.add(opt, "__next__")
        count_call = 0
        count_next = 0
        while 1:
            tsks, infos = self.sche.wait()
            if infos[0].method == "__call__":
                self.sche.add(opt, "__call__")
                count_call += 1
            elif infos[0].method == "__next__":
                self.sche.add(opt, "__next__")
                count_next += 1
            if count_call == 20 or count_next == 20:
                print(count_call, count_next)
                break
        t1 = time.time()
        print(t1-t0)
        



def suite():
    ray.init()
    suite = unittest.TestSuite()
    # suite.addTest(TestCase("test_sched_1_actor"))
    suite.addTest(TestCase("test_sche_opt"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法