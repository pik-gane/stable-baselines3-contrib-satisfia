import argparse
import warnings
from os import path

from experiments.custom_envs import ENV_DICT
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar


def load_model(log_path, name):
    model = ARDQN.load(path.join(log_path, "ARDQN", name, "models", "final_model"))
    model.name = name
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", type=str, required=True)
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    args = parser.parse_args()
    env = ENV_DICT[args.env_id]()
    models = []
    for name in args.names:
        if path.exists(path.join(args.log_path, "ARDQN", name, "models")):
            models.append(load_model(args.log_path, name))
        else:
            warnings.warn(f"Model {name} not found in {args.log_path}")
    print(f"Loaded {len(models)} models")
    fig = plot_ar(models, env=env, n_eval_episodes=args.n_eval_episodes)
    # Save plotly figure as html
    fig.write_html(path.join(args.log_path, "ARDQN", "results.html"))
