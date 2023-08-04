import os
import os
import pickle
import time
from os import path

import numpy as np
import pandas as pd
import ray
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import plotly.graph_objs as go

from custom_envs import MULTI_ARMED_BANDITS, ENV_DICT
from sb3_contrib.lra_dqn import LRADQN
from utils import open_tensorboard, DQNCallback

OPEN_TENSORBOARD = True
# end_name = input("Enter experiment end name: ")
EXPERIMENT_NAME = "LRADQN_exp_" + time.strftime("%Y%m%d-%H%M%S")
ray.init()
LEARNING_STEPS = 100_000
NB_LRA = 16
MODEL_PER_LRA = 10
ENV_ID = MULTI_ARMED_BANDITS
LOG_PATH = path.join("logs", EXPERIMENT_NAME, ENV_ID)
LOCAL_RELATIVE_ASPIRATION = list(np.linspace(0, 1, NB_LRA)) * MODEL_PER_LRA
TRAIN_DQN = True
N_EVAL_EPISODES = 100
MAKE_ENV = ENV_DICT[ENV_ID]

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
def train_model(lra, name):
    env = MAKE_ENV()
    if lra is None:
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
            save_path=path.join(LOG_PATH, "LRA", name, "models"),
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        model = LRADQN("MlpPolicy", env, lra, device="cpu")
        model.set_logger(tb_logger(path.join("LRA", name)))
        model.learn(LEARNING_STEPS, callback=[checkpoint_callback, DQNCallback()])
        model.save(path.join(LOG_PATH, "LRA", name, "models", f"final_model"))
        with open(path.join(LOG_PATH, "LRA", name, "results.pkl"), "wb") as f:
            pickle.dump(evaluate_policy(model, Monitor(env), n_eval_episodes=N_EVAL_EPISODES, deterministic=False), f)


print("Starting experiment " + EXPERIMENT_NAME)
names = [f"LRA_{round(a, 3)}_{i}" for i, a in enumerate(LOCAL_RELATIVE_ASPIRATION)]
names_iter = iter(names)
ray_models = [train_model.remote(lra, next(names_iter)) for lra in LOCAL_RELATIVE_ASPIRATION] + (
    [train_model.remote(None, None)] if TRAIN_DQN else []
)
try:
    ray.get(ray_models)
finally:
    ray.shutdown()


def load_model(name):
    model = LRADQN.load(path.join(LOG_PATH, "LRA", name, "models", "final_model"))
    model.name = name
    return model


def load_result(name):
    with open(path.join(LOG_PATH, "LRA", name, "results.pkl"), "rb") as f:
        return pickle.load(f)


results = []
models = []

for name in names:
    if path.exists(path.join(LOG_PATH, "LRA", name, "results.pkl")):
        model = load_model(name)
        if model.num_timesteps == LEARNING_STEPS:
            results.append(load_result(name))
            models.append(model)


df = pd.DataFrame(
    {
        "lra": [m.local_relative_aspiration for m in models],
        "gain_reward": [r[0] for r in results],
        "gain_std": [r[1] for r in results],
    }
)
# Group by lra and mean over the gain*
df = df.groupby("lra").mean().reset_index()
x = list(df["lra"])
y = df["gain_reward"]
y_upper = list(df["gain_reward"] + df["gain_std"])
y_lower = list(df["gain_reward"] - df["gain_std"])
fig = go.Figure(
    [
        go.Scatter(x=x, y=y, line=dict(color="rgb(0,100,80)"), mode="lines"),
        go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_upper + y_lower[::-1],  # upper, then lower reversed
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
        ),
    ]
)
# Add the y = x line in dashed black
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="rgb(0,0,0)", dash="dash")))

fig.show(renderer="browser")
