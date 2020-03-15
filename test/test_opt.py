import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import lmdb_op
from utils.dataloader import Dataloader
from agent import BasicWorker, DQN_Worker
from policy_optimizer import Optimizer
import torch

class TestCase(unittest.TestCase):
    def test_convergence(self):
        exc_worker = BasicWorker()
        buffer = "./ut_lmdb_l"
        lmdb_op.init(buffer)
        dataloader = Dataloader(buffer, lmdb_op, batch_size=256, worker_num=3, batch_num=10)
        opt = Optimizer(dataloader, iter_steps=10, update_period=1000)
        for i in range(600):
            traj, _ = exc_worker.__next__()
            lmdb_op.write(buffer, traj)
            print(i, lmdb_op.len(buffer))
        start = time.time()
        for i, loss in opt:
            print(i, loss)
            if i >= 10000:
                break
        print("time: {}".format(time.time()-start))
        # lmdb_op.clean(buffer)

    def test_train(self):
        exc_worker = DQN_Worker()
        buffer = "./ut_lmdb_l"
        lmdb_op.init(buffer)
        dataloader = Dataloader(buffer, lmdb_op, batch_size=64, worker_num=3, batch_num=40)
        opt = Optimizer(dataloader, iter_steps=400, update_period=10000)
        exc_worker.update(opt(), 1)
        count = 0
        while 1:
            traj, rw = next(exc_worker)
            lmdb_op.write(buffer, traj)
            if rw is not None:
                exc_worker.save("./train_video") if count % 100 == 0 else None
                print("worker reward: {} @ episod {}".format(rw, count))
                count += 1
            if lmdb_op.len(buffer) >= 100000:
                n_step, loss = next(opt)
                print("loss {} @ step {} with buff {}".format(loss, n_step, lmdb_op.len(buffer)))
                exc_worker.update(opt(), 0.05)
        lmdb_op.clean(buffer)
            

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    # suite.addTest(TestCase("test_convergence"))
    suite.addTest(TestCase("test_train"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法