import time
from os import path

from stable_baselines3.common.logger import configure

from public_good_envs import IteratedPD
# from custom_envs import MultiarmedBanditsEnv
from sb3_contrib import ARQLearning
from sb3_contrib.common.satisficing.evaluation import plot_ar, evaluate_policy
from utils import open_tensorboard

# from stable_baselines3 import DQN

OPEN_TENSORBOARD = 0
experiment = time.strftime("%Y%m%d-%H%M%S") + "_ARQlearningTest"
LEARNING_STEPS = 100
values = [0, 1, 2, 10]
variances = [0, 0, 0, 0]
nb_step = 10
env_id = 'MultiarmedBandits_' + '-'.join(str(i) for i in values) + f'_{nb_step}steps'


def make_env(obs_type=None, **kwargs):
    return IteratedPD(T=10, opponent="TitForTat")
    # return MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type, **kwargs)


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
aspiration = 15
verbose = 0
model = ARQLearning(env, aspiration, verbose=verbose, learning_rate=0.1,
                    mu=0.5)
# model.set_logger(tb_logger(path.join("ARQLearning", str(round(aspiration, 2)))))
model.learn(LEARNING_STEPS)  # , log_interval=10000000)
model.verbose = 0
plot_ar(env, [model])
