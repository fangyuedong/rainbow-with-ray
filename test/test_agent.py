import sys
import unittest 
import gym
import torch
import numpy as np
import cv2
sys.path.append("./")
from agent import DQN_Worker

class TestCase(unittest.TestCase):
 
    def test_agent_basic(self):
        for _ in range(4):
            worker = DQN_Worker()
            worker.update(None, 1)
            for _ in range(4):
                _ = next(worker)
                worker.save("./video")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_agent_basic"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法


