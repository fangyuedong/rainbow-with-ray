import os, sys
import torch
import torch.nn.functional as F
import gym
import cv2
import ray
sys.path.append("./")
from network.backbone import BasicNet
from network.dqn import DQN
from utils.dataloader import Dataloader, batch4net
from utils.atari_wrapper import wrap_rainbow

torch.backends.cudnn.benchmark = True

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""
f = lambda k, x: torch.from_numpy(x).cuda().float() if k != "action" else torch.from_numpy(x).cuda()

def smooth_l1_loss(x):
    return torch.where(x < 1, 0.5 * x ** 2, x - 0.5)

class Optimizer():
    def __init__(self, dataloader, env_name="PongNoFrameskip-v4", suffix="default", arch=DQN, backbone=BasicNet, 
        discount=0.99, update_period=10000, iter_steps=1, cuda=True, optimizer=torch.optim.Adam, **kwargs):
        assert isinstance(dataloader, Dataloader)
        self.dataloader = dataloader
        self.env = wrap_rainbow(gym.make(env_name), swap=True, phase="train")
        self.shape, self.na = self.env.observation_space.shape, self.env.action_space.n
        self.policy = arch(self.shape, self.na, backbone).train()
        self.target = arch(self.shape, self.na, backbone).train()
        self.target.load_state_dict(self.policy.state_dict())
        self.policy.cuda(), self.target.cuda() if cuda else None
        self.cuda = cuda
        self.prior = (self.dataloader.dataset._ray_actor_creation_function_descriptor.class_name == "Pmdb")
        if self.prior:
            self.beta = lambda x: 0.4+0.6*x/6.25e6 if "beta" not in kwargs else kwargs["beta"]
            self.max_IS = 0
        self.N = ray.get(self.dataloader.dataset.config.remote())["cap"]
        self.normalizer = torch.tensor(0).float().cuda()
        if optimizer == torch.optim.Adam:
            kwargs.update({"lr": 1e-4}) if "lr" not in kwargs else None
            kwargs.update({"weight_decay": 5e-5}) if "weight_decay" not in kwargs else None
            kwargs.update({"eps": 1.5e-4}) if "eps" not in kwargs else None
        self.optimizer = optimizer(self.policy.parameters(), **kwargs)
        self.iter_steps = iter_steps
        self.discount = discount
        self.update_period = update_period
        self.total_opt_steps = 0
        self.info = {}
        self.save_path = "./model/{}_{}/{}/{}".format(arch.__name__, backbone.__name__, env_name, suffix)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __iter__(self):
        return self

    def __next__(self, opt_steps=None, backprop=True):
        """iter and return policy params"""
        period = self.iter_steps if opt_steps == None else opt_steps
        sum_loss = 0
        for i, (data, check_id, idx, p) in enumerate(self.dataloader):
            data = batch4net(data, self.cuda)
            if self.prior:
                # IS = (self.N * torch.from_numpy(p).cuda()).pow(-self.beta(self.total_opt_steps))
                IS = (torch.from_numpy(p).cuda()).pow(-self.beta(self.total_opt_steps))
                # self.max_IS = IS.mean()
                # IS = IS / self.max_IS
                if self.total_opt_steps % 1000 == 0:
                    print(IS)
            else:
                IS = None
            loss, td_err = self.loss_fn(**data, IS=IS)
            if self.total_opt_steps % 1000 == 0:
                print(td_err)
            if self.prior:
                self.dataloader.update(idx, td_err.cpu().tolist(), check_id)
            if backprop:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.total_opt_steps += 1
            sum_loss += loss
            self.update_target() if self.total_opt_steps % self.update_period == 0 else None
            if i == period - 1:
                self.info["opt_steps"] = self.total_opt_steps
                self.info["loss"] = sum_loss.item() / (i+1)
                return self.info

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    @staticmethod
    def td_err(policy, target, state, action, next_state, reward, done, discount):
        with torch.no_grad():
            tar_v = discount * target.value(next_state) * (1 - done) + reward
        q_v = policy.value(state, action)
        assert q_v.shape == tar_v.shape
        return policy.td_err(q_v, tar_v)
    
    def loss_fn(self, state, action, next_state, reward, done, IS=None):
        err = self.td_err(self.policy, self.target, state, action, next_state, reward, done, self.discount)
        if IS is not None:
            loss = torch.mean(smooth_l1_loss(err) * IS)
        else:
            loss = torch.mean(smooth_l1_loss(err))
        return loss, err.detach()

    def __call__(self):
        """return policy params"""
        state = {}
        state["policy"] = self.policy.state_dict()
        state["target"] = self.target.state_dict()
        return state

    def save(self):
        path = os.path.join(self.save_path, "iter_{:0>6d}K.pkl".format(self.total_opt_steps))
        torch.save(self.policy.state_dict(), path)
        return path

    def config(self):
        info = {}
        info["batch_size"] = self.dataloader.batch_size
        info["iter_steps"] = self.iter_steps
        return info

class DQN_Opt(Optimizer):
    def __init__(self, dataloader, env_name="PongNoFrameskip-v4", suffix="default", arch=DQN, backbone=BasicNet, 
        discount=0.99, update_period=10000, iter_steps=1, cuda=True, optimizer=torch.optim.Adam, **kwargs):
        super(DQN_Opt, self).__init__(dataloader, env_name, suffix, arch, backbone, 
            discount, update_period, iter_steps, cuda, optimizer, **kwargs)   


class DDQN_Opt(DQN_Opt):
    def __init__(self, dataloader, env_name="PongNoFrameskip-v4", suffix="default", arch=DQN, backbone=BasicNet, 
        discount=0.99, update_period=10000, iter_steps=1, cuda=True, optimizer=torch.optim.Adam, **kwargs):
        super(DDQN_Opt, self).__init__(dataloader, env_name, suffix, arch, backbone, 
            discount, update_period, iter_steps, cuda, optimizer, **kwargs) 

    @staticmethod
    def td_err(policy, target, state, action, next_state, reward, done, discount):
        with torch.no_grad():
            act = policy.action(next_state)
            tar_v = discount * target.value(next_state, act) * (1 - done) + reward
        q_v = policy.value(state, action)
        assert q_v.shape == tar_v.shape
        return policy.td_err(q_v, tar_v)
        