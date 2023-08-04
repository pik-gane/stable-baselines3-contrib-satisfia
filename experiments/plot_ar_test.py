from itertools import product
from os import path
from stable_baselines3.common.envs import IdentityEnv
import numpy as np
from tqdm import tqdm

from sb3_contrib import ARDQN, ARQLearning
from sb3_contrib.common.satisficing.evaluation import plot_ar

env = IdentityEnv(2, ep_length=3)
aspirations = np.linspace(0, 1, 3)
mus = np.linspace(0, 1, 3)
rhos = np.linspace(0, 1, 3)


def make_model(aspiration, mu, rho):
    # model = ARDQN("MlpPolicy", env, initial_aspiration=aspiration, mu=mu, rho=rho, learning_starts=0, target_update_interval=100)
    model = ARQLearning(env, initial_aspiration=aspiration, mu=mu, rho=rho)
    # model.learn(2000)
    return model


models = [make_model(*args) for args in tqdm(product(aspirations, mus, rhos))]

plot = plot_ar(env, models, n_eval_episodes=5)
plot.show(renderer="browser")
