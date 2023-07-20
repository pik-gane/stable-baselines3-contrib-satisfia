import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
import torch as th
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from tqdm import tqdm

from sb3_contrib.ar_dqn import ARDQN
from sb3_contrib.common.satisficing.algorithms import ARQAlgorithm


def evaluate_policy(
    model: ARQAlgorithm,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = False,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Tuple[Union[Tuple[float, float], Tuple[List[float], List[int]]], Dict[str, np.ndarray]]:
    """
    Runs policy satisficing for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The satisficing RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions. Default to ``False``.
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
         # todo: doc
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((n_envs,), dtype=bool)
    model.switch_to_eval()
    lambdas = []
    aspirations = []
    reward_left = []
    with th.no_grad():
        current_lambdas = [[float(l)] for l in model.policy.lambda_ratio(observations, model.policy.initial_aspiration)]
    current_aspirations = [[model.policy.initial_aspiration] for _ in range(n_envs)]
    current_rew_left = [[model.policy.initial_aspiration] for _ in range(n_envs)]
    while (episode_counts < episode_count_targets).any():
        actions, _ = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        model.rescale_aspiration(observations, actions, rewards, new_observations, use_q_target=False)
        # logs
        new_aspiration = deepcopy(model.policy.aspiration).squeeze(1)
        with th.no_grad():
            new_lambda = model.policy.lambda_ratio(new_observations, model.policy.aspiration).squeeze(1).cpu().numpy()
        for i in range(n_envs):
            if not dones[i]:
                current_lambdas[i].append(new_lambda[i])
                current_aspirations[i].append(new_aspiration[i])
            current_rew_left[i].append(current_rew_left[i][-1] - rewards[i])

        model.reset_aspiration(dones)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            lambdas.append(current_lambdas[i])
                            aspirations.append(current_aspirations[i])
                            reward_left.append(current_rew_left[i])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        lambdas.append(current_lambdas[i])
                        aspirations.append(current_aspirations[i])
                        reward_left.append(current_rew_left[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    with th.no_grad():
                        current_lambdas[i] = [
                            float(model.policy.lambda_ratio(new_observations[i], model.policy.initial_aspiration))
                        ]
                    current_aspirations[i] = [model.policy.initial_aspiration]
                    current_rew_left[i] = [model.policy.initial_aspiration]
        observations = new_observations
        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mask = np.array([[i >= len(r) for i in range(max(map(len, reward_left)))] for r in reward_left])
    expand = lambda l: np.array([[x[i] if i < len(x) else x[-1] for i in range(max(map(len, l)))] for x in l])
    infos = {
        "lambda": np.ma.array(expand(lambdas), mask=mask[:, :-1]).mean(axis=0),
        "aspiration": np.ma.array(expand(aspirations), mask=mask[:, :-1]).mean(axis=0),
        "reward left": np.ma.array(expand(reward_left), mask=mask).mean(axis=0),
    }
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return (episode_rewards, episode_lengths), infos
    return (mean_reward, std_reward), infos


def plot_ar(
    models: List[ARQAlgorithm],
    results: Optional[List[Tuple[Tuple[float, float], Dict[str, np.ndarray]]]] = None,
    env: Optional[gym.Env] = None,
    n_eval_episodes: int = 100,
    **kwargs,
) -> Figure:
    """
    Plot the AR algorithm results.

    :param models: List of ARQAlgorithm models to plot
    :param results: List of results to plot, can be None if env is provided
    :param env: Environment to use for evaluation, can be None if results is provided
    :param n_eval_episodes: Number of episodes to use for evaluation

    :return: Plotly figure
    """
    if env is not None:
        eval_env = Monitor(env)
    else:
        assert results is not None, "Either env or results must be provided"

    # Create subplots
    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=(
            "Mean Lambda for each step",
            "Mean Aspiration Value for each step",
            "Mean Remaining return to go for each step",
            f"Mean Reward over {n_eval_episodes} episodes as a Function of Aspiration",
            f"Mean Reward over {n_eval_episodes} episodes as a Function of Rho",
            f"Mean Reward over {n_eval_episodes} episodes as a Function of Mu",
            "Standard Deviation of the Mean Reward compared to the Aspiration",
            "Maximum Deviation of the Mean Reward compared to the Aspiration",
        ),
    )
    fig.update_layout(height=1200)

    # create a continuous colorscale with len(models) colors that goes from green to red
    if len(models) > 1:
        colorscale = [
            f"rgba({int(255 * (1 - i / (len(models) - 1)))}, {int(255 * i / (len(models) - 1))}, 0, 1)"
            for i in range(len(models))
        ]
    else:
        colorscale = ["rgba(0, 255, 0,1)"]

    for i in tqdm(range(len(models))):
        model = models[i]
        # Check if model.name exists, else use model.policy.initial_aspiration
        try:
            model_name = model.name
        except AttributeError:
            model_name = str(round(model.policy.initial_aspiration, 2))
        if results is None:
            (m, std), infos = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, **kwargs)
        else:
            (m, std), infos = results[i]
        model.mean_reward = m
        model.std_reward = std
        model.i = i
        fig.add_trace(
            go.Scatter(
                y=infos["lambda"],
                name=model_name,
                line=dict(
                    color=colorscale[i],
                ),
                legendgroup=model_name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=infos["aspiration"],
                line=dict(
                    color=colorscale[i],
                ),
                showlegend=False,
                legendgroup=model_name,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                y=infos["reward left"],
                line=dict(
                    color=colorscale[i],
                ),
                showlegend=False,
                legendgroup=model_name,
            ),
            row=2,
            col=1,
        )

    aspirations_list = list(map(lambda x: x.policy.initial_aspiration, models))
    rho_list = list(map(lambda x: x.policy.rho, models))
    mu_list = list(map(lambda x: x.mu, models))

    def plot_reward(x, models, name, row, col, color):
        mean_rewards = np.array([m.mean_reward for m in models])
        std_rewards = np.array([m.std_reward for m in models])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_rewards,
                name=name,
                legendgroup=name,
                line=dict(
                    color=color,
                ),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_rewards + std_rewards,
                mode="lines",
                showlegend=False,
                line=dict(color="rgba(0,0,0,0)"),
                legendgroup=name,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_rewards - std_rewards,
                mode="lines",
                fill="tonexty",
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor=",".join(color.split(",")[:-1] + ["0.2)"]),
                showlegend=False,
                legendgroup=name,
            ),
            row=row,
            col=col,
        )

    mu_partitions = defaultdict(list)
    rho_partitions = defaultdict(list)
    aspiration_partitions = defaultdict(list)
    for model in models:
        mu_partitions[(model.policy.initial_aspiration, model.policy.rho)].append(model)
        rho_partitions[(model.policy.initial_aspiration, model.mu)].append(model)
        aspiration_partitions[(model.policy.rho, model.mu)].append(model)
    for (aspiration, mu), models in rho_partitions.items():
        rhos = list(map(lambda x: x.policy.rho, models))
        plot_reward(rhos, models, f"Mu:{round(mu,2)},Aspiration:{round(aspiration,2)}, share:", 3, 1, colorscale[models[0].i])
    for (aspiration, rho), models in mu_partitions.items():
        mus = list(map(lambda x: x.mu, models))
        plot_reward(mus, models, f"Rho:{round(rho,2)},Aspiration:{round(aspiration,2)},share:", 3, 2, colorscale[models[0].i])
    for (rho, mu), models in aspiration_partitions.items():
        aspirations = list(map(lambda x: x.policy.initial_aspiration, models))
        plot_reward(aspirations, models, f"Rho:{round(rho,2)},Mu:{round(mu,2)}, share:", 2, 2, colorscale[models[0].i])
    fig.add_trace(
        go.Scatter(
            x=sorted(list(set(aspirations_list))),
            y=sorted(list(set(aspirations_list))),
            mode="lines",
            line=dict(dash="dash", color="rgba(0,0,0,0.5)"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    # Add a heatmap of the deviation of the mean reward compared to the aspiration
    std_dev = {}
    max_std_dev = {}
    for (rho, mu), models in aspiration_partitions.items():
        mean_r = np.array([m.mean_reward for m in models])
        asps = np.array([m.policy.initial_aspiration for m in models])
        std_dev[(rho, mu)] = np.sqrt(np.square((mean_r - asps)).mean())
        max_std_dev[(rho, mu)] = np.sqrt(np.square((mean_r - asps)).max())
    fig.add_trace(
        go.Heatmap(
            z=[[std_dev[(rho, mu)] for rho in sorted(list(set(rho_list)))] for mu in sorted(list(set(mu_list)))],
            x=sorted(list(set(rho_list))),
            y=sorted(list(set(mu_list))),
            coloraxis="coloraxis",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=[[max_std_dev[(rho, mu)] for rho in sorted(list(set(rho_list)))] for mu in sorted(list(set(mu_list)))],
            x=sorted(list(set(rho_list))),
            y=sorted(list(set(mu_list))),
            coloraxis="coloraxis",
        ),
        row=4,
        col=2,
    )
    if isinstance(models[0], ARDQN):
        dev_per_share_mode = defaultdict(list)
        for model in models:
            dev_per_share_mode[model.policy.shared_network].append(model)
        for share_mode, models in dev_per_share_mode.items():
            dev_per_share_mode[share_mode] = np.sqrt(
                np.square(
                    np.array([m.mean_reward for m in models]) - np.array([m.policy.initial_aspiration for m in models])
                ).mean()
            )
        fig.add_trace(
            go.Bar(
                x=list(dev_per_share_mode.keys()),
                y=list(dev_per_share_mode.values()),
                marker=dict(color="rgba(0,0,0,0.5)"),
            ),
            row=5,
            col=1,
        )

    # Set coloraxis orientation ro horizontal and put it at the bottom of the plot and change color to reds
    fig.update_layout(coloraxis_colorbar=dict(orientation="h", y=-0.05, yanchor="top"), coloraxis=dict(colorscale="reds"))
    fig.update_layout(title_text=f"AR Plots for {n_eval_episodes} episodes")
    fig.update_yaxes(title_text="Lambda", row=1, col=1)
    fig.update_xaxes(title_text="Environment steps", row=1, col=1)
    fig.update_yaxes(title_text="Aspiration", row=1, col=2)
    fig.update_xaxes(title_text="Environment steps", row=1, col=2)
    fig.update_yaxes(title_text="Remaining return to go", row=2, col=1)
    fig.update_xaxes(title_text="Environment steps", row=2, col=1)
    fig.update_xaxes(title_text="Aspiration", row=2, col=2)
    fig.update_yaxes(title_text="Mean reward", row=2, col=2)
    fig.update_xaxes(title_text="Rho", row=3, col=1)
    fig.update_yaxes(title_text="Mean reward", row=3, col=1)
    fig.update_xaxes(title_text="Mu", row=3, col=2)
    fig.update_yaxes(title_text="Mean reward", row=3, col=2)
    fig.update_xaxes(title_text="Rho", row=4, col=1)
    fig.update_yaxes(title_text="Mu", row=4, col=1)
    fig.update_xaxes(title_text="Rho", row=4, col=2)
    fig.update_yaxes(title_text="Mu", row=4, col=2)
    fig.update_xaxes(title_text="Shared network", row=5, col=1)
    fig.update_yaxes(title_text="Mean std from aspiration", row=5, col=1)

    return fig


def evaluate_hard_policy(
    model: ARQAlgorithm,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = False,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Tuple[Union[Tuple[float, float], Tuple[List[float], List[int]]], Dict[str, np.ndarray]]:
    """
    Runs policy satisficing for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The satisficing RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions. Default to ``False``.
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
        # todo remove
         # todo: doc (same as evaluate_policy but with aspiration-= reward /= gamma
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((n_envs,), dtype=bool)
    model.switch_to_eval()
    lambdas = []
    aspirations = []
    reward_left = []
    with th.no_grad():
        current_lambdas = [[float(l)] for l in model.policy.lambda_ratio(observations, model.policy.initial_aspiration)]
    current_aspirations = [[model.policy.initial_aspiration] for _ in range(n_envs)]
    current_rew_left = [[model.policy.initial_aspiration] for _ in range(n_envs)]
    while (episode_counts < episode_count_targets).any():
        actions, _ = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        model.policy.aspiration -= rewards
        model.policy.aspiration /= model.policy.gamma
        # model.rescale_aspiration(observations, actions, new_observations, use_q_target=False)
        # logs
        new_aspiration = deepcopy(model.policy.aspiration)
        with th.no_grad():
            new_lambda = model.policy.lambda_ratio(new_observations, model.policy.aspiration).cpu().numpy()
        for i in range(n_envs):
            if not dones[i]:
                current_lambdas[i].append(new_lambda[i])
                current_aspirations[i].append(new_aspiration[i])
            current_rew_left[i].append(current_rew_left[i][-1] - rewards[i])

        model.reset_aspiration(dones)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            lambdas.append(current_lambdas[i])
                            aspirations.append(current_aspirations[i])
                            reward_left.append(current_rew_left[i])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        lambdas.append(current_lambdas[i])
                        aspirations.append(current_aspirations[i])
                        reward_left.append(current_rew_left[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    with th.no_grad():
                        current_lambdas[i] = [
                            float(model.policy.lambda_ratio(new_observations[i], model.policy.initial_aspiration))
                        ]
                    current_aspirations[i] = [model.policy.initial_aspiration]
                    current_rew_left[i] = [model.policy.initial_aspiration]

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    infos = {
        "lambda": np.array(lambdas).mean(axis=0),
        "aspiration": np.array(aspirations).mean(axis=0),
        "reward left": np.array(reward_left).mean(axis=0),
    }
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return (episode_rewards, episode_lengths), infos
    return (mean_reward, std_reward), infos
