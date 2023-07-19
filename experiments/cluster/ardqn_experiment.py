from os import path

import numpy as np

from experiments.custom_envs import DEFAULT_ASPIRATIONS, EMPTY_GRID, BOAT_RACE, MULTI_ARMED_BANDITS, PRISONERS
from slurm import submit_job_array
from itertools import product
import time

NB_ASPIRATION = 5
NB_RHO = 9
NB_MU = 9
EXPERIMENT_NAME = "ardqn_experiment_aspiration_rho_mu"
ENV_ID = PRISONERS
LOG_PATH = path.join("logs", EXPERIMENT_NAME, ENV_ID)
ASPIRATIONS = DEFAULT_ASPIRATIONS[ENV_ID](NB_ASPIRATION)
TRAIN_DQN = True
RHOS = np.linspace(0, 1, NB_RHO)
MUS = np.linspace(0, 1, NB_MU)
N_EVAL_EPISODES = 100

if __name__ == "__main__":
    # Compute the cartesian product of rhos mus and aspirations
    rhos, mus, aspirations = list(zip(*product(RHOS, MUS, ASPIRATIONS)))
    ar_names = [f"rho={rho}_mu={mu}_aspiration={aspiration}" for rho, mu, aspiration in zip(rhos, mus, aspirations)]
    nb_exp = len(aspirations) + TRAIN_DQN
    worker_args = {
        "aspirations": aspirations,
        "env_id": ENV_ID,
        "log_path": LOG_PATH,
        "learning_steps": 1_000_000,
        "rhos": rhos,
        "mus": mus,
        "policies": ["MlpPolicy"] * nb_exp,
        "names": ar_names + ["DQN"] * TRAIN_DQN,
        "n_eval_episodes": N_EVAL_EPISODES,
    }
    post_args = {
        "names": ar_names,
        "log_path": LOG_PATH,
    }
    submit_job_array(
        "ardqn_worker.py",
        worker_args,
        nb_exp,
        EXPERIMENT_NAME + "_" + ENV_ID,
        post_python_file="post_experiment_pickle.py",
        post_args=post_args,
        testing=False,
        wandb_sync=True,
    )
