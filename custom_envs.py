# Based on https://github.com/magni84/gym_bandits/blob/master/gym_bandits/envs/bandits_env.py
from typing import Optional
import numpy as np
import gymnasium as gym


class MultiarmedBanditsEnv(gym.Env):
    """Environment for multiarmed bandits"""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, values, deviations, render_mode: Optional[str] = None):
        assert len(values) == len(deviations), "values and deviations must have the same length"
        self.values = values
        self.deviations = deviations
        self.action_space = gym.spaces.Discrete(len(values))
        self.observation_space = gym.spaces.Discrete(1)
        self.optimal = np.argmax(self.values)
        self.state = 0
        self.time = 0

    def step(self, action: int):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward = self.np_random.standard_normal() * self.deviations[action] + self.values[action]
        self.time += 1
        return self.state, reward, self.time == 3, False, {"optimal": self.optimal}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.time = 0
        self.optimal = np.argmax(self.values)
        return self.state, {"optimal": self.optimal}

    def render(self, mode="human", close=False):
        return "You are playing a %d-armed bandit" % self.action_space.n
