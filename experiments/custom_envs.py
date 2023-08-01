# Based on https://github.com/magni84/gym_bandits/blob/master/gym_bandits/envs/bandits_env.py
from typing import Optional, Any

import pygame
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from experiments.public_good_envs import IteratedPD

try:
    from typing import Literal
except ImportError:  # Python <3.8 support:
    from typing_extensions import Literal

import gymnasium as gym
from gymnasium import spaces
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
            return min(self.time, self.nb_round - 1)
        elif self.obs_type == "one_hot":
            obs = np.zeros(self.nb_round, dtype=np.float32)
            obs[min(self.time, self.nb_round - 1)] = 1
            return obs
        else:
            return self.state

    def step(self, action: int):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        reward = self.np_random.standard_normal() * self.deviations[action] + self.values[action]
        self.time += 1
        return self.observation(), reward, self.time == self.nb_round, False, {"optimal": self.optimal}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.time = 0
        self.optimal = np.argmax(self.values)
        return self.observation(), {"optimal": self.optimal}

    def render(self, mode="human", close=False):
        return "You are playing a %d-armed bandit" % self.action_space.n


from ai_safety_gridworlds.environments.boat_race import BoatRaceEnvironment


class BoatRaceGymEnv(gym.Env):
    """A gym wrapper for the boat race environment."""

    def __init__(self, render_mode="pygame"):
        """Initialize the gym environment."""
        self._env = BoatRaceEnvironment()
        self._pygame_initialized = False
        self.render_mode = render_mode
        self.rows = 5
        self.cols = 5
        self.observation_space = gym.spaces.Discrete(5 * 5)
        self.action_space = gym.spaces.Discrete(4)
        self.timestep = self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Reset the environment and return the initial observation."""
        self.timestep = self._env.reset()
        return self._observation(self.timestep), {}

    def step(self, action):
        """Take an action and return the observation, reward, done and info."""
        self.timestep = self._env.step(action)
        obs = self._observation(self.timestep)
        reward = self.timestep.reward or 0
        done = self.timestep.last()
        info = {}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            if self.render_mode == "pygame":
                self._render_pygame()
            else:
                print(self.timestep[3])
        elif mode == "ansi":
            return str(self.timestep[3])
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Close the environment."""
        if self._pygame_initialized:
            pygame.quit()

    def _observation(self, timestep):
        """Convert the observation to a discrete value."""
        board = timestep.observation["board"]
        agent_position = np.where(board == 2)
        agent_row = agent_position[0][0]
        agent_col = agent_position[1][0]
        return agent_row * 5 + agent_col

    def _render_pygame(self):
        """Render the environment using pygame."""
        if not self._pygame_initialized:
            pygame.init()
            self._screen = pygame.display.set_mode((self.cols * 10, self.rows * 10))
            self._clock = pygame.time.Clock()
            self._pygame_initialized = True

        rgb = self.timestep[3]["RGB"]
        for row in range(self.rows):
            for col in range(self.cols):
                color = rgb[:, row, col]
                pygame.draw.rect(self._screen, color, (col * 10, row * 10, 10, 10))
        self._clock.tick(5)
        pygame.event.pump()
        pygame.display.flip()


def make_multi_armed_env():
    nb_step = 20
    values = np.array([0, 1, 2, 10]) / nb_step / 10
    variances = np.array([1, 1, 1, 1]) / nb_step / 10
    obs_type: Literal["step_count", "one_hot", "state"] = "step_count"
    env = MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type)
    return env


def make_boat_env():
    return BoatRaceGymEnv()


def make_empty_grid_env(render_mode="rgb_array", **kwargs):
    env = ImgObsWrapper(
        FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", max_episode_steps=100, render_mode=render_mode, **kwargs))
    )
    return env


def make_prisoners_dilemma_env(
    nb_rounds: int = 10, opponent: Literal["TitForTat", "STFT", "GTFT", "TFTT", "GrimTrigger", "Pavlov"] = "TitForTat"
):
    return IteratedPD(nb_rounds, opponent)


MULTI_ARMED_BANDITS = "MultiarmedBandits"
BOAT_RACE = "BoatRaceGymEnv"
EMPTY_GRID = "MiniGrid-Empty-5x5-v0"
PRISONERS = "PrisonersDilemma"

ENV_DICT = {
    MULTI_ARMED_BANDITS: make_multi_armed_env,
    BOAT_RACE: make_boat_env,
    EMPTY_GRID: make_empty_grid_env,
    PRISONERS: make_prisoners_dilemma_env,
}

DEFAULT_ASPIRATIONS = {
    MULTI_ARMED_BANDITS: lambda n: np.linspace(0, 1, num=n),
    BOAT_RACE: lambda n: np.linspace(-50, 50, num=n),
    EMPTY_GRID: lambda n: np.linspace(0, 1, num=n),
    PRISONERS: lambda n: np.linspace(0, 5, num=n),
}


class TimeTupleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.time_step = 0
        # Create a Tuple space that contains the original observation space and a Discrete space for the time
        self.observation_space = spaces.Tuple((env.observation_space, spaces.Discrete(env.spec.max_episode_steps)))

    def observation(self, observation):
        # Return a tuple that contains the original observation and the elapsed time
        return observation, self.time_step

    def reset(self, *, seed=None, options=None):
        # Reset the time counter
        self.time_step = 0
        # Call the reset method of the superclass and apply the observation method
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # Increment the time counter
        self.time_step += 1
        # Call the step method of the superclass and apply the observation method
        return super().step(action)

