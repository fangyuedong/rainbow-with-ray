import cv2
import numpy as np
from collections import deque
import gym
from gym import spaces


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Take action on reset for environments that are fixed until firing."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4, phase="train"):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        assert phase in ["train", "test"]
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip
        self.phase = phase
        self._rgbs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        done = None
        self._rgbs_buffer.clear()
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.phase == "test":
                self._rgbs_buffer.append(self.env.render(mode="rgb_array"))
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        self._rgbs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        if len(self._rgbs_buffer) > 0 and self.phase == "test":
            return np.max(np.stack(self._rgbs_buffer, axis=3), axis=3)
        else:
            return self.env.render(mode="rgb_array")

class FrameStackEnv(gym.Wrapper):
    def __init__(self, env, k):
        """Buffer observations and stack across channels (last axis)."""
        super(FrameStackEnv, self).__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        chn = env.observation_space.shape[-1] * k   
        self.observation_space = spaces.Box(low=0, high=255, shape=env.observation_space.shape[0:2]+(chn,))

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

class StackAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Stack skip frames together"""
        super(StackAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=skip)
        self._skip       = skip
        chn = env.observation_space.shape[-1] * skip     
        self.observation_space = spaces.Box(low=0, high=255, shape=env.observation_space.shape[0:2]+(chn,))

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            if not done:
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
            self._obs_buffer.append(obs)

        stack_obs = np.concatenate(self._obs_buffer, axis=2)
        return stack_obs, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        for _ in range(self._skip):
            self._obs_buffer.append(obs)
        stack_obs = np.concatenate(self._obs_buffer, axis=2)
        return stack_obs

def _process_frame84(frame):
    img = frame.astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 84),  interpolation=cv2.INTER_LINEAR)
    x_t = np.reshape(resized_screen, [84, 84, 1])
    return x_t.astype(np.uint8)

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def reset(self):
        return _process_frame84(self.env.reset())

class ClippedRewardsWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["reward"] = reward
        return obs, np.sign(reward), done, info

def _swap_chn(ob):
    return np.transpose(ob, (2,0,1))

class SwapChn(gym.Wrapper):
    def __init__(self, env=None):
        super(SwapChn, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.observation_space.shape[-1], )+self.observation_space.shape[:-1])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _swap_chn(obs), reward, done, info

    def reset(self):
        return _swap_chn(self.env.reset())


def wrap_deepmind_ram(env):
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, phase=phase)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClippedRewardsWrapper(env)
    return env

def wrap_deepmind(env, phase="train"):
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, phase=phase)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    if phase == "train":
        env = ClippedRewardsWrapper(env)
    return env

def wrap_rainbow(env, swap=False, phase="train"):
    assert 'NoFrameskip' in env.spec.id
    if phase == "train":
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = MaxAndSkipEnv(env, skip=4, phase=phase)
    env = FrameStackEnv(env, k=4)
    if phase == "train":
        env = ClippedRewardsWrapper(env)
    if swap == True:
        env = SwapChn(env)
    return env    

if __name__ == "__main__":
    import cv2
    env = gym.make("WizardOfWorNoFrameskip-v4")
    env = wrap_rainbow(env, swap=True, phase="test")
    print(env.observation_space)
    for _ in range(50):
        ob = env.reset()
        for _ in range(2000):
            ob, rw, done, info = env.step(env.action_space.sample())
            rgb = env.render(mode="rgb_array")
            cv2.imshow("true_video", rgb[:,:,::-1])
            cv2.waitKey(25)
            if done:
                break