# Will be called by slurm
import traceback
from os import path, environ

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import argparse
from experiments.custom_envs import ENV_DICT

from sb3_contrib import ARDQN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspirations", nargs="+", type=float, required=True)
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--learning_steps", type=int, required=True)
    parser.add_argument("--rhos", nargs="+", type=float, default=0.5)
    parser.add_argument("--mus", nargs="+", type=float, default=0.5)
    parser.add_argument("--policies", nargs="+", type=str, default="MlpPolicy")
    args = parser.parse_args()
    # Get slurm array task id
    task_id = int(environ["SLURM_ARRAY_TASK_ID"])
    aspiration = args.aspirations[task_id]
    rho = args.rhos[task_id]
    mu = args.mus[task_id]
    policy = args.policies[task_id]

    env = ENV_DICT[args.env_id]()

    # Create model
    model = ARDQN(policy, env, aspiration, verbose=0, rho=rho, mu=mu)
    # Setup logger
    tb_logger = configure(path.join(args.log_path, "ARDQN"), ["tensorboard"])
    model.set_logger(tb_logger)
    # Setup save callback
    save_path = path.join(args.log_path, "ARDQN", "models")
    callback = CheckpointCallback(save_freq=100_000, save_path=save_path)
    # Train model
    try:
        model.learn(args.learning_steps, callback=callback)
    except Exception as e:
        # If an exception was raised, add a line to the info file with the traceback
        with open(path.join(save_path, "info.txt")) as f:
            f.write(f"Exception raised:\n{traceback.format_exc()}\n\n")
        raise
    finally:
        # Save model
        model.save(path.join(save_path, f"final_model"))
        with open(path.join(save_path, "info.txt")) as f:
            f.write(f"aspiration: {aspiration}\n")
            f.write(f"rho: {rho}\n")
            f.write(f"mu: {mu}\n")
            f.write(f"policy: {policy}\n")
            f.write(f"learning_steps: {model.num_timesteps}\n")
