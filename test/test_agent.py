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
            for _ in range(4):
                data = next(worker)
                # for item in data:
                #     cv2.imshow("video", item["state"][0,:,:])
                #     cv2.waitKey(10)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_agent_basic"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法


