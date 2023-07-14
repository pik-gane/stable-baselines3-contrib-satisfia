# Will be called by slurm
import os

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from experiments.cluster.experiment import LEARNING_STEPS, RHO, MU, POLICY, ENV, ASPIRATIONS, EXPERIMENT_NAME, \
    LOG_PATH
from sb3_contrib import ARDQN

if __name__ == "__main__":
    # Get the aspiration from the slurm array index
    aspiration = ASPIRATIONS[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    # Get the experiment name from the slurm job name
    experiment_name = EXPERIMENT_NAME
    log_path = os.path.join(LOG_PATH, "ARDQN")
    # Create model
    model = ARDQN(POLICY, ENV, aspiration, verbose=0, rho=RHO, mu=MU)
    # Setup logger
    tb_logger = configure(log_path, ["tensorboard"])
    model.set_logger(tb_logger)
    # Setup save callback
    callback = CheckpointCallback(save_freq=100_000, save_path=os.path.join(log_path, "models"), name_prefix="model")
    # Train model
    model.learn(LEARNING_STEPS, callback=callback)
    # Save model
    model.save(os.path.join(log_path, "models", f"final_model_{LEARNING_STEPS}"))
