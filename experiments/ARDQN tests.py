import time
from itertools import product
from os import path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from tqdm import tqdm

from custom_envs import MultiarmedBanditsEnv, BoatRaceGymEnv
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard, DQNCallback

# from stable_baselines3 import DQN

OPEN_TENSORBOARD = False
experiment = time.strftime("%Y%m%d-%H%M%S") + "_ARDQNTest"
LEARNING_STEPS = 1000
nb_step = 30
values = np.array([0, 1, 2, 10]) / nb_step
variances = np.array([1, 1, 1, 1]) / nb_step
env_id = 'MultiarmedBandits_' + '-'.join(str(i) for i in values) + f'_{nb_step}steps'


def make_env(obs_type=None, **kwargs):
    return MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type, **kwargs)

# def make_env():
#     return BoatRaceGymEnv()

log_path = path.join("logs/tests", experiment)


# tmp_path = path.join("logs/old tests", experiment)


# set up logger
def tb_logger(exp):
    return configure(path.join(log_path, exp), ["tensorboard"])


def remove_logs():
    import shutil
    shutil.rmtree(log_path)


tb_window = None
if OPEN_TENSORBOARD:
    tb_window = open_tensorboard(log_path)

env = make_env(obs_type='step_count')
aspiration = 2
verbose = 0
# model = DQN("MlpPolicy", env, verbose=verbose, learning_rate=0.1, device='cpu', learning_starts=0)
models = []
specs = product(["none", "all", "features_extractor"], list(range(10)))
specs = [("all", 0)]
for share, i in tqdm(specs):
    model = ARDQN("MlpPolicy", env, aspiration, verbose=verbose, policy_kwargs=dict(shared_network=share), learning_starts=0)
    model.set_logger(tb_logger(path.join("ARDQN", share, str(i))))
    model.learn(LEARNING_STEPS, )  # , log_interval=10000000)
    models.append(model)
    model.name = f"{share}_{i}"
plot = plot_ar(models, env=env)
plot.show(renderer='browser')