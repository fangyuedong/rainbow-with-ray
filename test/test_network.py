import sys
import unittest 
import gym
import torch
import numpy as np
sys.path.append("./")
from network.dqn import DQN
from utils.atari_wrapper import wrap_rainbow

class TestCase(unittest.TestCase):
 
    def test_network_io_batch1(self):
        env = wrap_rainbow(gym.make("PongNoFrameskip-v4"), swap=True)
        ob = env.reset()
        assert ob.shape[0] == 4
        net = DQN(ob.shape, env.action_space.n).eval()

        torch_ob = torch.from_numpy(ob.astype(np.float32)).unsqueeze(0)
        o = net.forward(torch_ob)
        a = net.action(torch_ob)
        v = net.value(torch_ob, a=torch.tensor([0]))
        assert torch_ob.ndim - ob.ndim == o.ndim - 1
        assert torch_ob.ndim - ob.ndim == a.ndim
        assert torch_ob.ndim - ob.ndim == v.ndim
 
    def test_network_io_batch2(self):
        env = wrap_rainbow(gym.make("PongNoFrameskip-v4"), swap=True)
        ob = env.reset()
        assert ob.shape[0] == 4
        net = DQN(ob.shape, env.action_space.n).eval()

        torch_ob = torch.from_numpy(ob.astype(np.float32)).unsqueeze(0)
        torch_ob = torch.cat((torch_ob, torch_ob), dim=0)
        o = net.forward(torch_ob)
        a = net.action(torch_ob)
        v = net.value(torch_ob, a=torch.tensor([0, 0]))
        assert torch_ob.ndim - ob.ndim == o.ndim - 1
        assert torch_ob.ndim - ob.ndim == a.ndim
        assert torch_ob.ndim - ob.ndim == v.ndim

    def test_network_io_batch0(self):
        env = wrap_rainbow(gym.make("PongNoFrameskip-v4"), swap=True)
        ob = env.reset()
        assert ob.shape[0] == 4
        net = DQN(ob.shape, env.action_space.n).eval()

        torch_ob = torch.from_numpy(ob.astype(np.float32))
        o = net.forward(torch_ob)
        a = net.action(torch_ob)
        v = net.value(torch_ob, a=torch.tensor(0))
        assert torch_ob.ndim - ob.ndim == o.ndim - 1
        assert torch_ob.ndim - ob.ndim == a.ndim
        assert torch_ob.ndim - ob.ndim == v.ndim

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCase("test_network_io_batch1"))
    suite.addTest(TestCase("test_network_io_batch2"))
    suite.addTest(TestCase("test_network_io_batch0"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法


