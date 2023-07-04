import threading
import time
from os import path

import numpy as np
import ray
from stable_baselines3.common.logger import configure

from custom_envs import MultiarmedBanditsEnv
from sb3_contrib import ARQLearning
from sb3_contrib.common.satisficing.evaluation import plot_ar
from utils import open_tensorboard

OPEN_TENSORBOARD = True

ray.init()
experiment = time.strftime("%Y%m%d-%H%M%S")
user_input = ""


def get_user_input():
    global user_input
    user_input = input("Experiment subname (can be empty): ")


# Start the input thread
input_thread = threading.Thread(target=get_user_input)
input_thread.start()
# Wait for 30 seconds for user input
input_thread.join(timeout=30)
experiment = "_".join([experiment, user_input]) if user_input else experiment
LEARNING_STEPS = 10_000
values = [0, 1, 2, 10]
variances = [0, 0, 0, 0]
nb_step = 3
obs_type = "step_count"
env_id = "MultiarmedBandits_" + "-".join(str(i) for i in values) + f"_{nb_step}steps" + "_" + obs_type
log_path = path.join("./logs/tests", experiment, env_id)


# set up logger
def tb_logger(exp):
    return configure(path.join(log_path, exp), ["tensorboard"])


tb_window = None
if OPEN_TENSORBOARD:
    tb_window = open_tensorboard(log_path)


@ray.remote
def train_model(aspiration, env, log_path, learning_steps, tb_logger):
    model = ARQLearning(env, policy_kwargs=dict(initial_aspiration=aspiration), gamma=1)
    model.set_logger(tb_logger(path.join("ArQLearning", str(round(aspiration, 2)))))
    model.learn(learning_steps, progress_bar=True)
    model.save(path.join(log_path, "ArQLearning", str(round(aspiration, 2)), "models", str(learning_steps)))
    return aspiration


aspirations = np.linspace(1, nb_step * max(values), num=11)
print("Starting experiment " + experiment)
env = MultiarmedBanditsEnv(values, variances, nb_step, obs_type=obs_type)
ray_models = [train_model.remote(a, env, log_path, LEARNING_STEPS, tb_logger) for a in aspirations]
ray.get(ray_models)
ray.shutdown()
models = list(
    map(lambda a: ARQLearning.load(path.join(log_path, "ArQLearning", str(round(a, 2)), "models", str(LEARNING_STEPS))),
        aspirations))
plot_ar(env, models)
