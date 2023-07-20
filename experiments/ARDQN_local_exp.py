import os
import threading
import time
from os import path
from typing import Literal

import gymnasium as gym
import numpy as np
import ray
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from custom_envs import MultiarmedBanditsEnv, BoatRaceGymEnv, make_boat_env
from sb3_contrib import ARDQN, QLearning, ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard, DQNCallback

OPEN_TENSORBOARD = True
USE_DQN = False
end_name = input("Enter experiment end name: ")
experiment = "ARDQN_exp" + (end_name if end_name else time.strftime("%Y%m%d-%H%M%S"))
ray.init()


LEARNING_STEPS = 1_000_000
nb_aspiration = 15



make_env = make_boat_env
env, env_id, aspirations = make_env(init=True)
log_path = path.join("logs", experiment, env_id)
# Check if log path already exists
if path.exists(log_path):
    # Show what is in the folder
    print("Log folder already exists. Content:")
    for f in os.listdir(log_path):
        print(f)
    # Ask if we want to overwrite
    if input("Do you want to overwrite? (y/n)") != "y":
        exit()


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
            save_freq=100_000,
            save_path=path.join(log_path, "DQN", "models"),
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model = DQN("MlpPolicy", env, learning_starts=1000)
        model.set_logger(tb_logger("DQN"))
        model.learn(learning_steps, callback=[DQNCallback(), checkpoint_callback])
        model.save(path.join(log_path, "DQN", "models", str(learning_steps)))
    else:
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=path.join(log_path, "ARDQN", str(round(aspiration, 2)), "models"),
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
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
