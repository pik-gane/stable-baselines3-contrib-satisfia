from os import path

import numpy as np

from experiments.custom_envs import DEFAULT_ASPIRATIONS, EMPTY_GRID, BOAT_RACE, MULTI_ARMED_BANDITS
from slurm import submit_job_array
from itertools import product
import time

NB_ASPIRATION = 5
NB_RHO = 9
NB_MU = 9
EXPERIMENT_NAME = "ardqn_experiment_aspiration_rho_mu"
ENV_ID = EMPTY_GRID
LOG_PATH = path.join("logs", EXPERIMENT_NAME, ENV_ID)
ASPIRATIONS = DEFAULT_ASPIRATIONS[ENV_ID](NB_ASPIRATION)
TRAIN_DQN = True
RHOS = np.linspace(0, 1, NB_RHO)
MUS = np.linspace(0, 1, NB_MU)

if __name__ == "__main__":
    # Compute the cartesian product of rhos mus and aspirations
    rhos, mus, aspirations = list(zip(*product(RHOS, MUS, ASPIRATIONS)))
    names = [f"rho={rho}_mu={mu}_aspiration={aspiration}" for rho, mu, aspiration in zip(rhos, mus, aspirations)]
    names += ["DQN"] * TRAIN_DQN
    nb_exp = len(aspirations) + TRAIN_DQN
    worker_args = {
        "aspirations": aspirations,
        "env_id": ENV_ID,
        "log_path": LOG_PATH,
        "learning_steps": 1_000_000,
        "rhos": rhos,
        "mus": mus,
        "policies": ["MlpPolicy"] * nb_exp,
        "names": names,
    }
    post_args = {
        "names": names,
        "env_id": ENV_ID,
        "log_path": LOG_PATH,
    }
    submit_job_array(
        "ardqn_worker.py",
        worker_args,
        nb_exp,
        EXPERIMENT_NAME,
        "post_experiment.py",
        post_args=post_args,
        testing=False,
        wandb_sync=True,
    )
