import numpy as np
import torch

'''
    ob -> value -> action -> reward -> done -> 
'''
class PriorTrace():
    def __init__(self, discount=0.99):
        # self.prior_list = []
        # self.value, self.action, self.reward = None, None, None
        # self.next_stage = "value"
        self.reinit()
        self.discount = discount

    def add_value(self, value):
        assert self.next_stage == "value"
        if self.value is None:
            self.value = value
            # print(self.value)
        else:
            prior = self.value[self.action] - (self.reward + self.discount * value.max())
            prior = prior.abs()
            self.prior_list.append(prior)
            self.value = value
        self.next_stage = "action"

    def add_action(self, action):
        assert self.next_stage == "action"
        self.action = action
        # print(self.action)
        self.next_stage = "reward"

    def add_reward(self, reward):
        assert self.next_stage == "reward"
        self.reward = reward
        # print(self.reward)
        self.next_stage = "done"

    def add_done(self, done):
        assert self.next_stage == "done"
        # print(done)
        if done:
            prior = self.value[self.action] - self.reward
            prior = prior.abs()
            self.prior_list.append(prior)
            self.value, self.action, self.reward = None, None, None
        self.next_stage = "value"

    def prior(self):
        return torch.stack(self.prior_list).cpu().tolist()

    def reinit(self):
        self.prior_list = []
        self.value, self.action, self.reward = None, None, None
        self.next_stage = "value"        