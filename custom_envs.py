# Based on https://github.com/magni84/gym_bandits/blob/master/gym_bandits/envs/bandits_env.py
from typing import Optional, Literal

import gymnasium as gym
import numpy as np


class MultiarmedBanditsEnv(gym.Env):
    """Environment for multiarmed bandits"""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        values,
        deviations,
        nb_round,
        render_mode: Optional[str] = None,
        obs_type: Literal["step_count", "one_hot", "state"] = "state",
    ):
        assert len(values) == len(deviations), "values and deviations must have the same length"
        self.values = values
        self.deviations = deviations
        self.nb_round = nb_round
        self.obs_type = obs_type
        self.action_space = gym.spaces.Discrete(len(values))
        if obs_type == "state":
            self.observation_space = gym.spaces.Discrete(1)
        elif obs_type == "one_hot":
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(nb_round,), dtype=np.float32)
        elif obs_type == "step_count":
            self.observation_space = gym.spaces.Discrete(nb_round)
        else:
            raise NotImplementedError(f"Unknown observation type: {obs_type}")
        self.optimal = np.argmax(self.values)
        self.state = 0
        self.time = 0

    def observation(self):
        if self.obs_type == "step_count":
            return self.time
        elif self.obs_type == "one_hot":
            obs = np.zeros(self.nb_round, dtype=np.float32)
            if self.time < self.nb_round:
                obs[self.time] = 1
            return obs
        else:
            return self.state

    def step(self, action: int):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward = self.np_random.standard_normal() * self.deviations[action] + self.values[action]
        self.time += 1
        assert self.time <= self.nb_round, "Too many steps"
        return self.observation(), reward, self.time == self.nb_round, False, {"optimal": self.optimal}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.time = 0
        self.optimal = np.argmax(self.values)
        return self.observation(), {"optimal": self.optimal}

    def render(self, mode="human", close=False):
        return "You are playing a %d-armed bandit" % self.action_space.n
