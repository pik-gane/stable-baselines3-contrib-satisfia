# Will be called by slurm
import argparse
import pickle
import traceback
from os import path, environ

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from experiments.custom_envs import ENV_DICT
from experiments.utils import DQNCallback
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import evaluate_policy as ar_evaluate_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspirations", nargs="+", type=float, required=True)
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--learning_steps", type=int, required=True)
    parser.add_argument("--names", nargs="+", type=str, required=True)
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.5])
    parser.add_argument("--mus", nargs="+", type=float, default=[0.5])
    parser.add_argument("--policies", nargs="+", type=str, default=["MlpPolicy"])
    parser.add_argument("--shared_network", nargs="+", type=str, default=["none"])
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    args = parser.parse_args()
    # Get slurm array task id
    task_id = int(environ["SLURM_ARRAY_TASK_ID"]) if args.task_id is None else args.task_id
    name = args.names[task_id]
    env = ENV_DICT[args.env_id]()
    if task_id >= len(args.aspirations):
        # Run DQN instead
        policy = args.policies[task_id]
        model = DQN(policy, env, verbose=0)
        # Setup logger
        log_path = path.join(args.log_path, "DQN")
        tb_logger = configure(log_path, ["tensorboard"])
        model.set_logger(tb_logger)
        # Setup save callback
        save_path = path.join(log_path, "models")
        save_callback = CheckpointCallback(save_freq=100_000, save_path=save_path)
        # Train model
        try:
            model.learn(args.learning_steps, callback=[save_callback, DQNCallback()])
        except Exception as e:
            with open(path.join(save_path, "error.txt"), "w") as f:
                f.write(f"Exception raised:\n{traceback.format_exc()}\n\n")
            raise
        finally:
            # Save model
            model.save(path.join(save_path, f"final_model"))
            with open(path.join(save_path, "info.txt"), "w") as f:
                f.write(f"policy: {policy}\n")
                f.write(f"learning_steps: {model.num_timesteps}\n")
        # Evaluate model and save results in a text file
        with open(path.join(log_path, "results.txt"), "w") as f:
            f.write(f"reward: {evaluate_policy(model, env, n_eval_episodes=args.n_eval_episodes)}\n")
    else:
        aspiration = args.aspirations[task_id]
        rho = args.rhos[task_id]
        mu = args.mus[task_id]
        policy = args.policies[task_id]
        shared_network = args.shared_network[task_id]
        # Create model
        model = ARDQN(policy, env, aspiration, verbose=0, rho=rho, mu=mu, policy_kwargs=dict(shared_network=shared_network))
        # Setup logger
        log_path = path.join(args.log_path, "ARDQN", name)
        tb_logger = configure(log_path, ["tensorboard"])
        model.set_logger(tb_logger)
        # Setup save callback
        save_path = path.join(log_path, "models")
        callback = CheckpointCallback(save_freq=100_000, save_path=save_path)
        # Train model
        try:
            model.learn(args.learning_steps, callback=callback)
        except Exception as e:
            # If an exception was raised, add a line to the info file with the traceback
            with open(path.join(save_path, "error.txt"), "w") as f:
                f.write(f"Exception raised:\n{traceback.format_exc()}\n\n")
            raise
        finally:
            # Save model
            model.save(path.join(save_path, f"final_model"))
            with open(path.join(save_path, "info.txt"), "w") as f:
                f.write(f"aspiration: {aspiration}\n")
                f.write(f"rho: {rho}\n")
                f.write(f"mu: {mu}\n")
                f.write(f"policy: {policy}\n")
                f.write(f"learning_steps: {model.num_timesteps}\n")
        # Evaluate model and save results in a pickle file
        with open(path.join(log_path, "results.pkl"), "wb") as f:
            pickle.dump(ar_evaluate_policy(model, env, n_eval_episodes=args.n_eval_episodes), f)
