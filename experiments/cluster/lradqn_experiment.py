from itertools import product
from os import path

import numpy as np

from experiments.custom_envs import EMPTY_GRID
from slurm import submit_job_array

NB_LRA = 20
MODEL_PER_LRA = 10
EXPERIMENT_NAME = "lra_experiment" + input("Complete the experiment name (or press enter to use default): lra_experiment")
ENV_ID = EMPTY_GRID
LOG_PATH = path.join("logs", EXPERIMENT_NAME, ENV_ID)
LOCAL_RELATIVE_ASPIRATIONS = list(np.linspace(0, 1, NB_LRA))
TRAIN_DQN = True
N_EVAL_EPISODES = 100
LEARNING_STEPS = 1_000_000

if __name__ == "__main__":
    p = list(product(LOCAL_RELATIVE_ASPIRATIONS, range(MODEL_PER_LRA)))
    lra_names = [f"LRA_{lra:.3f}_{i}" for lra, i in p]
    nb_exp = NB_LRA * MODEL_PER_LRA + TRAIN_DQN
    worker_args = {
        "lras": [x[0] for x in p],
        "env_id": ENV_ID,
        "log_path": LOG_PATH,
        "learning_steps": LEARNING_STEPS,
        "policies": ["MlpPolicy"] * nb_exp,
        "names": lra_names + ["DQN"] * TRAIN_DQN,
        "n_eval_episodes": N_EVAL_EPISODES,
    }
    post_args = {"names": lra_names, "log_path": LOG_PATH, "expected_time_steps": LEARNING_STEPS}
    submit_job_array(
        "lradqn_worker.py",
        worker_args,
        nb_exp,
        EXPERIMENT_NAME + "_" + ENV_ID,
        post_python_file="lradqn_post_experiment.py",
        post_args=post_args,
        testing=False,
        wandb_sync=True,
    )
