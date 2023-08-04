import os
import pickle
import time
from itertools import product
from os import path

import ray
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from tqdm import tqdm

from experiments.custom_envs import *
from sb3_contrib.common.satisficing.evaluation import evaluate_policy as evaluate_ar_policy
from sb3_contrib.common.satisficing.evaluation import plot_ar
from sb3_contrib.common.satisficing.utils import ratio
from sb3_contrib.lra_dqn import LRADQN
from sb3_contrib.lra_dqn.lrar_dqn import LRARDQN
from utils import open_tensorboard

OPEN_TENSORBOARD = False
experiment = "LRADQNTest" + time.strftime("%Y%m%d-%H%M%S")  # + input("Experiment name: ")
LEARNING_STEPS = 100_000
ENV_ID = MULTI_ARMED_BANDITS
nb_round = 30


# def make_env():
#     # return PublicGood(nb_rounds=10, n_players=10, alpha=2, sigma=1)
#     return IteratedPD(nb_rounds=nb_round, opponent="GTFT")


make_env = ENV_DICT[ENV_ID]

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
verbose = 0
lra_min = 0.1
lra_max = 0.9
model_min = LRADQN(
    "MlpPolicy",
    make_env(),
    lra_min,
    verbose=verbose,
    device="cpu",
)
model_min.learn(LEARNING_STEPS)
print("Min model learned")
model_max = LRADQN(
    "MlpPolicy",
    make_env(),
    lra_max,
    verbose=verbose,
    device="cpu",
)
model_max.learn(LEARNING_STEPS)
print("Max model learned")
print(
    f"Min model result: {evaluate_policy(model_min, make_env(), n_eval_episodes=100)}\n"
    f"Max model result: {evaluate_policy(model_max, make_env(), n_eval_episodes=100)}"
)


rho = np.linspace(0, 1, 20)
aspirations = DEFAULT_ASPIRATIONS[ENV_ID](20)
p = list(product(rho, aspirations))
models = [LRARDQN(asp, rho, model_min, model_max) for rho, asp in p]


class DummyPolicy:
    def __init__(self, model: LRARDQN):
        self.initial_aspiration = model.initial_aspiration
        self.rho = model.rho
        self.model = model

    @property
    def aspiration(self):
        return self.model.aspiration

    def lambda_ratio(self, obs, aspirations):
        q = self.model.q(obs, aspirations)
        q_min = q.min(dim=1).values
        q_max = q.max(dim=1).values
        lambdas = ratio(q_min, th.tensor(aspirations, device=self.model.device), q_max)
        lambdas[q_max == q_min] = 0.5  # If q_max == q_min, we set lambda to 0.5, this should not matter
        return lambdas.clamp(min=0, max=1)


tmp_path = path.join("/tmp", experiment)
os.makedirs(tmp_path, exist_ok=True)
for model in models:
    model.mu = 0  # Add a dummy mu for plot_ar
    model.name = f"rho: {round(model.rho, 2)}, asp: {model.aspiration.round(2)}"
    model.switch_to_eval = lambda: None
    model.policy = DummyPolicy(model)

# ray.init(_plasma_directory="/tmp")
ray.init()


@ray.remote
def eval_model(model):
    with open(path.join(tmp_path, model.name), "wb") as f:
        pickle.dump(evaluate_ar_policy(model, make_env(), n_eval_episodes=100), f)


for i in tqdm(range(len(models) // 16)):
    ray.get([eval_model.remote(model) for model in models[i * 16 : (i + 1) * 16]])
ray.get([eval_model.remote(model) for model in models[(i + 1) * 16 :]])


def load_results(model):
    with open(path.join(tmp_path, model.name), "rb") as f:
        return pickle.load(f)


try:
    fig = plot_ar(models, results=[load_results(model) for model in models])
    fig.show(renderer="browser")
    os.makedirs(log_path, exist_ok=True)
    fig.write_html(path.join(log_path, "results.html"))
finally:
    ray.shutdown()
