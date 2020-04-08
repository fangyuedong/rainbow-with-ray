import sys
import unittest 
import gym
import torch
import numpy as np
import cv2
import time
sys.path.append("./")
from agent import DQN_Worker

class TestCase(unittest.TestCase):
 
    def test_agent_train(self):
        for _ in range(4):
            worker = DQN_Worker(env_name="AsterixNoFrameskip-v4")
            worker.update(None, 1)
            lenth = 0
            t0 = time.time()
            for _ in range(20):
                info = next(worker)
                lenth += info["episod_len"]
            print("{}fps".format(lenth/(time.time()-t0)))
    
    def test_agent_test(self):
        worker = DQN_Worker(env_name="AsterixNoFrameskip-v4", phase="test")
        for _ in range(20):
            info = next(worker)
            print(info)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_agent_train"))
    suite.addTest(TestCase("test_agent_test"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法


