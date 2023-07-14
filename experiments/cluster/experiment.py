from os import path

from experiments.custom_envs import make_boat_env
from slurm import submit_job_array

NB_ASPIRATION = 15
LEARNING_STEPS = 1_000_000
MAKE_ENV = make_boat_env
RHO = 0.0
MU = 0.5
POLICY = "MlpPolicy"
ENV, ENV_ID, ASPIRATIONS = MAKE_ENV(init=True, nb_aspiration=NB_ASPIRATION)
EXPERIMENT_NAME = f"ARDQN_exp_rho{RHO}_mu{MU}"
LOG_PATH = path.join("logs", EXPERIMENT_NAME, ENV_ID)

if __name__ == "__main__":
    submit_job_array("job_worker.py", NB_ASPIRATION, EXPERIMENT_NAME, "post_experiment.py")
