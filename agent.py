# from network.backbone import BasicNet
# from network.dqn import DQN
from atari_wrapper import wrap_rainbow
import gym
import cv2
import random

"""
Transition = {"state": np.array, "action": int, "next_state": np.array, "reward": float, "done": logical}
"""

class BasicWorker():
    def __init__(self, env_name="PongNoFrameskip-v4", workid="work_{:0>2d}".format(0), max_steps=100000, debug=False):
        self.env = wrap_rainbow(gym.make("PongNoFrameskip-v4"))
        self.workid = workid
        self.env_name = "PongNoFrameskip-v4"
        self.debug = debug
        self.max_steps = max_steps
        print("{}\t{}\tActions: {}".format(self.workid, self.env_name, self.env.action_space.n))

    def exc(self):
        cache, done, count = [], False, 0
        ob = self.env.reset()
        while count < self.max_steps and not done:
            act = self._action()
            next_ob, rw, done, _ = self.env.step(act)
            if self.debug:
                cv2.imshow("video", ob[:,:,0])
                cv2.waitKey(25)
            cache.append({"state": ob, "action": act, "next_state": next_ob, "reward": rw, "done": done})
            ob = next_ob
            count += 1
        return cache
    
    def _action(self):
        return self.env.action_space.sample()
    
    # def update(self):
    #     action = random.randint(0, self.env.action_space.n-1)
    #     def func():
    #         return action
    #     self._action = func

# class Agent(BasicWorker):
#     def __init__(self, env_name="PongNoFrameskip-v4", alg=DQN, backbone=BasicNet)

if __name__ == "__main__":
    worker = BasicWorker(debug=True)
    while 1:
        worker.exc()