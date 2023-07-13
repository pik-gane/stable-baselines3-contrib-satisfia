import threading
import time
from os import path

import numpy as np
import ray
from stable_baselines3.common.logger import configure

from custom_envs import MultiarmedBanditsEnv, BoatRaceGymEnv
from sb3_contrib import ARQLearning, QLearning
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard

OPEN_TENSORBOARD = True
Q_LEARNING = False

experiment = time.strftime("%Y%m%d-%H%M%S")
user_input = ""
user_want_custom_name = False


def get_user_input():
    global user_want_custom_name
    i = input("Do you want to set a custom experiment name? (empty for no, anything else for yes): ")
    user_want_custom_name = True if i else False


# Start the input thread
input_thread = threading.Thread(target=get_user_input)
input_thread.start()
# Wait for 30 seconds for user input
input_thread.join(timeout=30)
if user_want_custom_name:
    user_input = input("Experiment subname (can be empty): ")
ray.init()
experiment = "_".join([experiment, user_input]) if user_input else experiment
# values = [0, 1, 2, 10]
# variances = [1, 1, 1, 1]
# nb_step = 20
# obs_type = "step_count"
# env_id = "MultiarmedBandits_" + "-".join(f"{values[i]}_{variances[i]}" for i in range(len(values))) + f"_{nb_step}steps" + "_" + obs_type
# env = MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type)


def make_env():
    return BoatRaceGymEnv()


env_id = "BoatRaceGymEnv"

LEARNING_STEPS = 1_000_000
nb_aspiration = 64
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
        model = QLearning(env, gamma=1, learning_rate=0.1)
        model.set_logger(tb_logger("QLearning"))
        model.learn(learning_steps)
        model.save(path.join(log_path, "QLearning", "models", str(learning_steps)))
    else:
        model = ARQLearning(env, policy_kwargs=dict(initial_aspiration=aspiration), learning_rate=0.1, mu=0.5)
        model.set_logger(tb_logger(path.join("ARQLearning", str(round(aspiration, 2)))))
        model.learn(learning_steps)
        model.save(path.join(log_path, "ARQLearning", str(round(aspiration, 2)), "models", str(learning_steps)))


# aspirations = np.linspace(min(values) * nb_step, nb_step * max(values), num=nb_aspiration)
aspirations = np.linspace(-100, 50, num=nb_aspiration)
print("Starting experiment " + experiment)
ray_models = [train_model.remote(a, make_env, log_path, LEARNING_STEPS, tb_logger) for a in aspirations] + (
    [train_model.remote(None, make_env, log_path, LEARNING_STEPS, tb_logger)] if Q_LEARNING else []
)
try:
    ray.get(ray_models)
finally:
    ray.shutdown()
models = list(
    map(
        lambda a: ARQLearning.load(path.join(log_path, "ARQLearning", str(round(a, 2)), "models", str(LEARNING_STEPS))),
        aspirations,
    )
)
fig = plot_ar(make_env(), models)
# Save plotly figure as html
fig.write_html(path.join(log_path, "ARQLearning", "results.html"))
