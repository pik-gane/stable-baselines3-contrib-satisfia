import time
from os import path

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from custom_envs import MultiarmedBanditsEnv
from sb3_contrib import ARQLearning
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard

# from stable_baselines3 import DQN

OPEN_TENSORBOARD = False
experiment = time.strftime("%Y%m%d-%H%M%S")
LEARNING_STEPS = 10_000
values = [0, 1, 2, 10]
variances = [0, 0, 0, 0]
nb_step = 3
env_id = 'MultiarmedBandits_' + '-'.join(str(i) for i in values) + f'_{nb_step}steps'


def make_env(obs_type=None, **kwargs):
    return MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type, **kwargs)


# tmp_path = path.join("./logs/tests", experiment)
tmp_path = path.join("/logs/old tests", experiment)


# set up logger
def tb_logger(exp):
    return configure(path.join(tmp_path, exp), ["tensorboard"])


tb_window = None
if OPEN_TENSORBOARD:
    tb_window = open_tensorboard(tmp_path)

env = make_env(obs_type='step_count')
aspiration = 30
model = ARQLearning(env, verbose=2, policy_kwargs=dict(initial_aspiration=aspiration), gamma=1, learning_rate=0.1)
# model.set_logger(tb_logger("QLearning"))
try:
    model.learn(LEARNING_STEPS, progress_bar=True, log_interval=10000000)
finally:
    model.verbose = 0
    print(evaluate_policy(model, Monitor(env), n_eval_episodes=100, render=False))
    plot_ar(env, [model])
