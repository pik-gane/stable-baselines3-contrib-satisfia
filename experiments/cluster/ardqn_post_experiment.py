import argparse
from os import path

from experiments.custom_envs import ENV_DICT
from sb3_contrib import ARDQN
from sb3_contrib.common.satisficing.evaluation import plot_ar
import pickle


def load_model(log_path, name):
    model = ARDQN.load(path.join(log_path, "ARDQN", name, "models", "final_model"))
    model.name = name
    return model


def load_results(log_path, name):
    with open(path.join(log_path, "ARDQN", name, "results.pkl"), "rb") as f:
        results = pickle.load(f)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    args = parser.parse_args()
    models = list(
        map(
            load_model,
            args.names,
        )
    )
    results = list(
        map(
            load_results,
            args.names,
        )
    )

    fig = plot_ar(models, results=results)
    # Save plotly figure as html
    fig.write_html(path.join(args.log_path, "ARDQN", "results.html"))
