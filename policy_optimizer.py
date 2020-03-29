import sys
import torch
import gym
import cv2
sys.path.append("./")
from network.backbone import BasicNet
from network.dqn import DQN
from utils.dataloader import Dataloader
from utils.atari_wrapper import wrap_rainbow

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""
f = lambda k, x: torch.from_numpy(x).cuda().float() if k != "action" else torch.from_numpy(x).cuda()

class Optimizer():
    def __init__(self, dataloader, env_name="PongNoFrameskip-v4", arch=DQN, backbone=BasicNet, 
        discount=0.99, update_period=10000, iter_steps=1, cuda=True, optimizer=torch.optim.Adam, **kwargs):
        assert isinstance(dataloader, Dataloader)
        self.dataloader = dataloader
        self.env = wrap_rainbow(gym.make("PongNoFrameskip-v4"), swap=True, phase="train")
        self.shape, self.na = self.env.observation_space.shape, self.env.action_space.n
        self.policy = arch(self.shape, self.na, backbone).train()
        self.target = arch(self.shape, self.na, backbone).train()
        self.target.load_state_dict(self.policy.state_dict())
        self.policy.cuda(), self.target.cuda() if cuda else None

        kwargs.update({"lr": 1e-4}) if "lr" not in kwargs else None
        kwargs.update({"weight_decay": 5e-5}) if "weight_decay" not in kwargs else None
        self.optimizer = optimizer(self.policy.parameters(), **kwargs)
        self.iter_steps = iter_steps
        self.discount = discount
        self.update_period = update_period
        self.total_opt_steps = 0
        self.info = {}

        
    def __iter__(self):
        return self

    def __next__(self, opt_steps=None):
        """iter and return policy params"""
        period = self.iter_steps if opt_steps == None else opt_steps
        sum_loss = 0
        for i, data in enumerate(self.dataloader):
            data = {k: f(k, v) for k, v in data.items()}
            loss = self.loss_fn(**data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_opt_steps += 1
            sum_loss += loss.item()
            self.update_target() if self.total_opt_steps % self.update_period == 0 else None
            if i == period - 1:
                self.info["opt_steps"] = self.total_opt_steps
                self.info["loss"] = sum_loss / (i+1)
                return self.info

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def loss_fn(self, state, action, next_state, reward, done):
        with torch.no_grad():
            target = self.discount * self.target.value(next_state) * (1 - done) + reward
        q_fn = self.policy.value(state, action)
        assert q_fn.shape == target.shape
        loss = self.policy.loss_fn(q_fn, target)
        return loss

    def __call__(self):
        """return policy params"""
        return self.policy.state_dict()

class DQN_Opt(Optimizer):
    def __init__(self, dataloader, env_name="PongNoFrameskip-v4", arch=DQN, backbone=BasicNet, 
        discount=0.99, update_period=10000, iter_steps=1, cuda=True, optimizer=torch.optim.Adam, **kwargs):
        super(DQN_Opt, self).__init__(dataloader, env_name, arch, backbone, 
            discount, update_period, iter_steps, cuda, optimizer, **kwargs)


class DDQN_Opt(DQN_Opt):
    def __init__(self, dataloader, env_name="PongNoFrameskip-v4", arch=DQN, backbone=BasicNet, 
        discount=0.99, update_period=10000, iter_steps=1, cuda=True, optimizer=torch.optim.Adam, **kwargs):
        super(DDQN_Opt, self).__init__(dataloader, env_name, arch, backbone, 
            discount, update_period, iter_steps, cuda, optimizer, **kwargs) 

    def loss_fn(self, state, action, next_state, reward, done):
        with torch.no_grad():
            act = self.policy.action(next_state)
            target = self.discount * self.target.value(next_state, act) * (1 - done) + reward
        q_fn = self.policy.value(state, action)
        assert q_fn.shape == target.shape
        loss = self.policy.loss_fn(q_fn, target)
        return loss