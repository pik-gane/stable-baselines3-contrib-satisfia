from os import path

from experiments.custom_envs import DEFAULT_ASPIRATIONS
from slurm import submit_job_array

NB_ASPIRATION = 20
EXPERIMENT_NAME = "ardqn_experiment"
LOG_PATH = path.join("logs", EXPERIMENT_NAME)
ENV_ID = "MiniGrid-Empty-5x5-v0"
ASPIRATIONS = DEFAULT_ASPIRATIONS[ENV_ID](NB_ASPIRATION)

if __name__ == "__main__":
    worker_args = {
        "aspirations": ASPIRATIONS,
        "env_id": ENV_ID,
        "log_path": LOG_PATH,
        "learning_steps": 1_000_000,
        "rhos": [0.0] * NB_ASPIRATION,
        "mus": [0.5] * NB_ASPIRATION,
        "policies": ["MlpPolicy"] * NB_ASPIRATION,
    }
    post_args = {
        "aspirations": ASPIRATIONS,
        "env_id": ENV_ID,
        "log_path": LOG_PATH,
    }
    submit_job_array("ardqn_worker.py", worker_args, NB_ASPIRATION, EXPERIMENT_NAME, "post_experiment.py", post_args=post_args)
