import argparse
from os import path

from experiments.custom_envs import ENV_DICT
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspirations", nargs="+", type=float, required=True)
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    args = parser.parse_args()
    env = ENV_DICT[args.env_id]()
    models = list(
        map(
            lambda a: ARDQN.load(
                path.join(args.log_path, "ARDQN", str(round(a, 2)), "models", "final_model")
            ),
            args.aspirations,
        )
    )
    fig = plot_ar(env, models)
    # Save plotly figure as html
    fig.write_html(path.join(args.log_path, "ARDQN", "results.html"))
