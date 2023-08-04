import time
from itertools import product
from os import path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from tqdm import tqdm

from experiments.public_good_envs import IteratedPD
from public_good_envs import PublicGood

# from custom_envs import MultiarmedBanditsEnv, BoatRaceGymEnv
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard, DQNCallback

# from stable_baselines3 import DQN

OPEN_TENSORBOARD = False
experiment = time.strftime("%Y%m%d-%H%M%S") + "_ARDQNTest"  # + input("Experiment name: ")
LEARNING_STEPS = 1_000
nb_round = 10


def make_env():
    # return PublicGood(nb_rounds=10, n_players=10, alpha=2, sigma=1)
    return IteratedPD(nb_rounds=nb_round, opponent="GTFT")


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

env = make_vec_env(make_env, n_envs=10)
# env = make_env()
aspiration = 0.5
verbose = 0
# model = DQN("MlpPolicy", env, verbose=verbose, learning_rate=0.1, device='cpu', learning_starts=0)
models = []
specs = list(product(["none", "all", "features_extractor", "min_max"], [True, False], [0]))
for share, use_delta_net, i in tqdm(specs):
    model = ARDQN(
        "MlpPolicy",
        env,
        aspiration,
        verbose=verbose,
        policy_kwargs=dict(shared_network=share),
        rho=1,
        learning_starts=0,
        device="cpu",
        target_update_interval=100,
        use_delta_nets=use_delta_net,
    )
    model.set_logger(tb_logger(path.join("ARDQN", share, str(i))))
    model.learn(LEARNING_STEPS)  # , log_interval=10000000)
    models.append(model)
    model.name = f"{share}_{i}"
plot = plot_ar(models, env=make_env())
plot.show(renderer="browser")
