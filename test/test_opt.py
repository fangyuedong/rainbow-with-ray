import sys, os
import unittest 
import time
import ray
import random
import numpy as np
sys.path.append("./")
from utils.replay_buffer import mmdb_op as db_op
from utils.dataloader import Dataloader
from agent import BasicWorker, DQN_Worker
from policy_optimizer import DDQN_Opt as Optimizer
import torch
import pickle

# def load_pkl():
#     path = "test_convengence.pkl"
#     if os.path.exists(path):
#         return pkl.load(path)
#     else:
#         return None
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
                exc_worker.save("./train_video") if count % 100 == 0 else None
                print("worker reward: {} @ episod {}".format(wk_info["episod_rw"], count))
                count += 1
            if db_op.len(buffer) >= 10000:
                opt_info = next(opt)
                print("loss {} @ step {} with buff {}".format(opt_info["loss"], opt_info["opt_steps"], db_op.len(buffer)))
                exc_worker.update(opt(), 0.05)
                if opt_info["opt_steps"] == 10000:
                    break
        db_op.clean(buffer)

    def test_convengence(self):
        buffer = db_op.init("./ut_lmdb_l")
        if not os.path.exists("data.pkl"):
            data = []
            exc_worker = DQN_Worker(db=buffer, db_write=db_op.write)    
            exc_worker.update(None, 1.0)
            while db_op.len(buffer) < 1000000:
                next(exc_worker)
                print(db_op.len(buffer))
            for i in range(1000000):
                data += db_op.read(buffer, i, decompress=False)
            with open("data.pkl", "wb") as fo:
                pickle.dump(data, fo)
        else:
            with open("data.pkl", "rb") as fo:
                data = pickle.load(fo, encoding='bytes')
            db_op.write(buffer, data, compress=False)

        dataloader = Dataloader(buffer, db_op, batch_size=256, worker_num=3, batch_num=5)  
        opt = Optimizer(dataloader, iter_steps=5, update_period=10000, lr=0.625e-4)
        while 1:
            opt_info = next(opt)
            print("loss {} @ step {} with buff {}".format(opt_info["loss"], opt_info["opt_steps"], db_op.len(buffer)))
 
#提供名为suite()的全局方法，PyUnit在执行测试的过程调用suit()方法来确定有多少个测试用例需要被执行，
#可以将TestSuite看成是包含所有测试用例的一个容器。
def suite():
    ray.init()
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_train"))
    suite.addTest(TestCase("test_convengence"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法