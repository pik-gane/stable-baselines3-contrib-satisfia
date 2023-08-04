import argparse
import pickle
import warnings
from os import path

import pandas as pd
import plotly.graph_objects as go

from sb3_contrib import LRADQN


def load_model(log_path, name):
    model = LRADQN.load(path.join(log_path, "LRADQN", name, "models", "final_model"))
    model.name = name
    return model


def load_results(log_path, name):
    with open(path.join(log_path, "LRADQN", name, "results.pkl"), "rb") as f:
        results = pickle.load(f)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--expected_time_steps", type=int, required=True)

    args = parser.parse_args()
    models = []
    results = []
    for name in args.names:
        if path.exists(path.join(args.log_path, "LRADQN", name, "results.pkl")):
            model = load_model(args.log_path, name)
            if model.num_timesteps == args.expected_time_steps:
                results.append(load_results(args.log_path, name))
                models.append(model)
            else:
                warnings.warn(
                    f"Model {name} has {model.num_timesteps} timesteps, expected {args.expected_time_steps}\n"
                    "and will therefore not be plotted."
                )
        else:
            warnings.warn(f"Model {name} not found in {args.log_path}")

    df = pd.DataFrame(
        {
            "lra": [m.local_relative_aspiration for m in models],
            "gain_reward": [r[0] for r in results],
            "gain_std": [r[1] for r in results],
        }
    )
    # Group by lra and mean over the gain*
    df = df.groupby("lra").mean().reset_index()
    x = list(df["lra"])
    y = df["gain_reward"]
    y_upper = list(df["gain_reward"] + df["gain_std"])
    y_lower = list(df["gain_reward"] - df["gain_std"])
    fig = go.Figure(
        [
            go.Scatter(x=x, y=y, line=dict(color="rgb(0,100,80)"), mode="lines"),
            go.Scatter(
                x=x + x[::-1],  # x, then x reversed
                y=y_upper + y_lower[::-1],  # upper, then lower reversed
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            ),
        ]
    )
    # Add the y = x line in dashed black
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="rgb(0,0,0)", dash="dash")))
    # Save plotly figure as html
    fig.write_html(path.join(args.log_path, "LRADQN", "results.html"))
