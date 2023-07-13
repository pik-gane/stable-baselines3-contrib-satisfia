import threading
import time
from os import path

import gymnasium as gym
import numpy as np
import ray
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from custom_envs import MultiarmedBanditsEnv, BoatRaceGymEnv
from sb3_contrib import ARDQN, QLearning, ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard, DQNCallback

OPEN_TENSORBOARD = True
USE_DQN = False

experiment = time.strftime("%Y%m%d-%H%M%S") + "_ARDQN_exp" + input("Enter experiment end name: ")
ray.init()


env_id = ""
LEARNING_STEPS = 1_000_000
nb_aspiration = 15
aspirations = None

def make_multi_armed_env(init=False):
    values = np.array([0, 1, 2, 10]) / 10
    variances = np.array([1, 1, 1, 1]) / 10
    nb_step = 10
    obs_type = "step_count"
    if init:
        global env_id, aspirations
        aspirations = np.linspace(min(values) * nb_step, nb_step * max(values), num=nb_aspiration)
        env_id = (
            "MultiarmedBandits_"
            + "-".join(f"{values[i]}_{variances[i]}" for i in range(len(values)))
            + f"_{nb_step}steps"
            + "_"
            + obs_type
        )
    return MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type)


def make_boat_env(init=False):
    if init:
        global env_id, aspirations
        aspirations = np.linspace(-50, 50, num=nb_aspiration)
        env_id = "BoatRaceGymEnv"
    return BoatRaceGymEnv()


def make_empty_grid_env(init=False, render_mode="rgb_array", **kwargs):
    if init:
        global env_id, aspirations
        aspirations = np.linspace(0, 1, num=nb_aspiration)
        env_id = "MiniGrid-Empty-5x5-v0"
    else:
        return ImgObsWrapper(
            FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", max_episode_steps=100, render_mode=render_mode, **kwargs))
        )


make_env = make_empty_grid_env
env = make_env(init=True)
log_path = path.join("logs", experiment, env_id)


# set up logger
def tb_logger(exp):
    return configure(path.join(log_path, exp), ["tensorboard"])


tb_window = None
if OPEN_TENSORBOARD:
    tb_window = open_tensorboard(log_path)


@ray.remote
def train_model(aspiration, make_env, log_path, learning_steps, tb_logger):
    env = make_env()
    if aspiration is None:
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=path.join(log_path, "DQN", "models"),
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        model = DQN("MlpPolicy", env, learning_starts=1000)
        model.set_logger(tb_logger("DQN"))
        model.learn(learning_steps, callback=[DQNCallback(), checkpoint_callback])
        model.save(path.join(log_path, "DQN", "models", str(learning_steps)))
    else:
        checkpoint_callback = CheckpointCallback(
            save_freq=1500_000,
            save_path=path.join(log_path, "ARDQN", str(round(aspiration, 2)), "models", str(learning_steps)),
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        model = ARDQN("MlpPolicy", env, aspiration, rho=0)
        model.set_logger(tb_logger(path.join("ARDQN", str(round(aspiration, 2)))))
        model.learn(learning_steps, callback=[checkpoint_callback])
        model.save(path.join(log_path, "ARDQN", str(round(aspiration, 2)), "models", str(learning_steps)))


# aspirations = np.linspace(min(values) * nb_step, nb_step * max(values), num=nb_aspiration)
# aspirations = np.linspace(0, 50, num=nb_aspiration)
print("Starting experiment " + experiment)
ray_models = [train_model.remote(a, make_env, log_path, LEARNING_STEPS, tb_logger) for a in aspirations] + (
    [train_model.remote(None, make_env, log_path, LEARNING_STEPS, tb_logger)] if USE_DQN else []
)
try:
    ray.get(ray_models)
finally:
    ray.shutdown()
models = list(
    map(
        lambda a: ARDQN.load(path.join(log_path, "ARDQN", str(round(a, 2)), "models", str(LEARNING_STEPS))),
        aspirations,
    )
)
fig = plot_ar(make_env(), models)
# Save plotly figure as html
fig.write_html(path.join(log_path, "ARDQN", "results.html"))
