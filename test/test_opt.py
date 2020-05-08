import sys
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import pmdb_op as db_op
from utils.dataloader import Dataloader
from agent import BasicWorker, DQN_Worker
from policy_optimizer import Optimizer
import torch

class TestCase(unittest.TestCase):

    def test_train(self):
        buffer = db_op.init("./ut_lmdb_l")
        exc_worker = DQN_Worker(db=buffer, db_write=db_op.write)
        dataloader = Dataloader(buffer, db_op, batch_size=64, worker_num=3, batch_num=40)
        opt = Optimizer(dataloader, iter_steps=400, update_period=10000)
        exc_worker.update(opt(), 1)
        count = 0
        while 1:
            wk_info = next(exc_worker)
            if wk_info is not None:
                # exc_worker.save("./train_video") if count % 100 == 0 else None
                print("worker reward: {} @ episod {}".format(wk_info["episod_rw"], count))
                count += 1
            if db_op.len(buffer) >= 10000:
                opt_info = next(opt)
                print("loss {} @ step {} with buff {}".format(opt_info["loss"], opt_info["opt_steps"], db_op.len(buffer)))
                exc_worker.update(opt(), 0.05)
                if opt_info["opt_steps"] == 10000:
                    break
        db_op.clean(buffer)
            

    
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_train"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法