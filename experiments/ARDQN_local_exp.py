import itertools
import os
import threading
import time
from itertools import product
from os import path
from typing import Literal
import pickle
import gymnasium as gym
import numpy as np
import ray
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from custom_envs import DEFAULT_ASPIRATIONS, PRISONERS, MULTI_ARMED_BANDITS, ENV_DICT, EMPTY_GRID
from sb3_contrib import ARDQN, QLearning, ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar
from sb3_contrib.common.satisficing.evaluation import evaluate_policy as ar_evaluate_policy
from stable_baselines3.common.evaluation import evaluate_policy
from utils import open_tensorboard, DQNCallback

OPEN_TENSORBOARD = True
end_name = input("Enter experiment end name: ")
EXPERIMENT_NAME = "ARDQN_exp_" + (end_name if end_name else time.strftime("%Y%m%d-%H%M%S"))
ray.init()

LEARNING_STEPS = 1_000_000
NB_ASPIRATION = 10
NB_RHO = 1
NB_MU = 5
# SHARE = ["none", "all", "features_extractor"]
SHARE = ["none"]
ENV_ID = MULTI_ARMED_BANDITS
LOG_PATH = path.join("logs", EXPERIMENT_NAME, ENV_ID)
# ASPIRATIONS = DEFAULT_ASPIRATIONS[ENV_ID](NB_ASPIRATION)
ASPIRATIONS = [0.5] * NB_ASPIRATION
TRAIN_DQN = False
RHOS = np.linspace(0, 1, NB_RHO)
MUS = np.linspace(0, 1, NB_MU)
N_EVAL_EPISODES = 100
MAKE_ENV = ENV_DICT[ENV_ID]
params = list(product(RHOS, MUS, ASPIRATIONS, SHARE))

# Check if log path already exists
if path.exists(LOG_PATH):
    # Show what is in the folder
    print("Log folder already exists. Content:")
    for file in os.listdir(LOG_PATH):
        print(file)
    # Ask if we want to overwrite
    if input("Do you want to overwrite? (y/n)") != "y":
        ray.shutdown()
        exit()


# set up logger
def tb_logger(exp):
    return configure(path.join(LOG_PATH, exp), ["tensorboard"])


tb_window = None
if OPEN_TENSORBOARD:
    tb_window = open_tensorboard(LOG_PATH)


@ray.remote
def train_model(rho, mu, aspiration, share, name):
    env = MAKE_ENV()
    if aspiration is None:
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=path.join(LOG_PATH, "DQN", "models"),
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model = DQN("MlpPolicy", env, learning_starts=1000)
        model.set_logger(tb_logger("DQN"))
        model.learn(LEARNING_STEPS, callback=[DQNCallback(), checkpoint_callback])
        model.save(path.join(LOG_PATH, "DQN", "models", f"final_model"))
    else:
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=path.join(LOG_PATH, "ARDQN", name, "models"),
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model = ARDQN("MlpPolicy", env, aspiration, rho=rho, mu=mu, policy_kwargs=dict(shared_network=share), device="cpu")
        model.set_logger(tb_logger(path.join("ARDQN", name)))
        model.learn(LEARNING_STEPS, callback=[checkpoint_callback])
        model.save(path.join(LOG_PATH, "ARDQN", name, "models", f"final_model"))
        with open(path.join(LOG_PATH, "ARDQN", name, "results.pkl"), "wb") as f:
            pickle.dump(ar_evaluate_policy(model, Monitor(env), n_eval_episodes=N_EVAL_EPISODES), f)


print("Starting experiment " + EXPERIMENT_NAME)
names = [f"{i}_mu_{mu}_rho_{rho}_asp_{asp}_share_{share}" for i, (rho, mu, asp, share) in enumerate(params)]
names_iter = iter(names)
ray_models = [train_model.remote(rho, mu, aspiration, share, next(names_iter)) for rho, mu, aspiration, share in params] + ([
    train_model.remote(None, None, None, None)
] if TRAIN_DQN else [])
try:
    ray.get(ray_models)
finally:
    ray.shutdown()


def load_model(name):
    model = ARDQN.load(path.join(LOG_PATH, "ARDQN", name, "models", "final_model"))
    model.name = name
    return model


def load_result(name):
    with open(path.join(LOG_PATH, "ARDQN", name, "results.pkl"), "rb") as f:
        return pickle.load(f)


results = []
models = []

for name in names:
    if path.exists(path.join(LOG_PATH, "ARDQN", name, "results.pkl")):
        model = load_model(name)
        if model.num_timesteps == LEARNING_STEPS:
            results.append(load_result(name))
            models.append(model)

fig = plot_ar(models, results=results)
# Save plotly figure as html
fig.write_html(path.join(LOG_PATH, "ARDQN", "results.html"))
fig.show(renderer="browser")
