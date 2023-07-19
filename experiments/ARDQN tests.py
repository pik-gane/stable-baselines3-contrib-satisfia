import time
from os import path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from public_good_envs import PublicGood
#from custom_envs import MultiarmedBanditsEnv, BoatRaceGymEnv
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard, DQNCallback

# from stable_baselines3 import DQN

OPEN_TENSORBOARD = False
experiment = time.strftime("%Y%m%d-%H%M%S") + "_ARDQNTest"
LEARNING_STEPS = 100
values = np.array([0, 1, 2, 10]) / 10
variances = np.array([1, 1, 1, 1]) / 10
nb_step = 10
env_id = 'MultiarmedBandits_' + '-'.join(str(i) for i in values) + f'_{nb_step}steps'


def make_env(obs_type=None, **kwargs):
    return PublicGood(nb_rounds=10, n_players=10, alpha=2, sigma=1)
#    return MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type, **kwargs)

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
aspiration = 5
verbose = 0
# model = DQN("MlpPolicy", env, verbose=verbose, learning_rate=0.1, device='cpu', learning_starts=0)
model = ARDQN("MlpPolicy", env, aspiration, verbose=verbose,)
model_dqn = DQN("MlpPolicy", env, verbose=verbose, learning_starts=0)
model.learn(LEARNING_STEPS, callback=[DQNCallback()])
model.set_logger(tb_logger(path.join("ARDQN", str(round(aspiration, 2)))))
model.learn(LEARNING_STEPS)  # , log_interval=10000000)
model.verbose = 0
plot = plot_ar(env, [model])
