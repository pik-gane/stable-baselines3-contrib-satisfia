import time
from itertools import product
from os import path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from tqdm import tqdm

from experiments.custom_envs import make_multi_armed_env
from experiments.public_good_envs import IteratedPD
from public_good_envs import PublicGood

# from custom_envs import MultiarmedBanditsEnv, BoatRaceGymEnv
from sb3_contrib import ARDQN
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.lra_dqn import LRADQN
from utils import open_tensorboard, DQNCallback

from stable_baselines3 import DQN

OPEN_TENSORBOARD = True
experiment = "LRADQNTest" + time.strftime("%Y%m%d-%H%M%S")  # + input("Experiment name: ")
LEARNING_STEPS = 50_000
nb_round = 30


def make_env():
    # return PublicGood(nb_rounds=10, n_players=10, alpha=2, sigma=1)
    return IteratedPD(nb_rounds=nb_round, opponent="GTFT")


make_env = make_multi_armed_env

# def make_env():
#     return BoatRaceGymEnv()

log_path = path.join("logs/tests", experiment)

# set up logger
def tb_logger(exp):
    return configure(path.join(log_path, exp), ["tensorboard"])


def remove_logs():
    import shutil

    shutil.rmtree(log_path)


tb_window = None
if OPEN_TENSORBOARD:
    tb_window = open_tensorboard(log_path)

# env = make_vec_env(make_env, n_envs=10)
env = make_env()
verbose = 0
# model = DQN("MlpPolicy", env, verbose=verbose, learning_rate=0.1, device='cpu', learning_starts=0)
models = []
local_relative_aspiration = 0.5
model = LRADQN(
    "MlpPolicy",
    env,
    local_relative_aspiration,
    verbose=verbose,
    learning_starts=0,
    device="cpu",
    target_update_interval=100,
)
model.set_logger(tb_logger("LRA_DQN"))
model.learn(LEARNING_STEPS, callback=DQNCallback())
print(f'result: {evaluate_policy(model, make_env(), deterministic=False)}')

