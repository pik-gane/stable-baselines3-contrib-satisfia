from typing import List, Union

import gymnasium as gym
import numpy as np
import pytest
import torch as th
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib import ARDQN, QRDQN, TQC, MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.utils import get_action_masks


class FlattenBatchNormDropoutExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input and applies batch normalization and dropout.
    Used as a placeholder when feature extraction is not needed.
    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super().__init__(
            observation_space,
            get_flattened_obs_dim(observation_space),
        )
        self.flatten = nn.Flatten()
        self.batch_norm = nn.BatchNorm1d(self._features_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        result = self.flatten(observations)
        result = self.batch_norm(result)
        result = self.dropout(result)
        return result


def clone_batch_norm_stats(batch_norm: nn.BatchNorm1d) -> (th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the given batch norm layer.
    :param batch_norm:
    :return: the bias and running mean
    """
    return batch_norm.bias.clone(), batch_norm.running_mean.clone()


def clone_qrdqn_batch_norm_stats(model: QRDQN) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the quantile network and quantile-target network.
    :param model:
    :return: the bias and running mean from the quantile network and quantile-target network
    """
    quantile_net_batch_norm = model.policy.quantile_net.features_extractor.batch_norm
    quantile_net_bias, quantile_net_running_mean = clone_batch_norm_stats(quantile_net_batch_norm)

    quantile_net_target_batch_norm = model.policy.quantile_net_target.features_extractor.batch_norm
    quantile_net_target_bias, quantile_net_target_running_mean = clone_batch_norm_stats(quantile_net_target_batch_norm)

    return quantile_net_bias, quantile_net_running_mean, quantile_net_target_bias, quantile_net_target_running_mean


def clone_ar_ddqn_batch_norm_stats(model: ARDQN) -> List[th.Tensor]:
    """
    Clone the bias and running mean from the quantile network and quantile-target network.
    :param model:
    :return: the bias and running mean from the quantile network and quantile-target network
    """
    tensors = []
    for net in [
        model.policy.q_net,
        model.policy.delta_qmax_net,
        model.policy.delta_qmin_net,
        model.policy.q_net_target,
    ]:
        net_batch_norm = net.features_extractor.batch_norm
        net_bias, net_running_mean = clone_batch_norm_stats(net_batch_norm)
        tensors += [net_bias, net_running_mean]
    return tensors


def clone_tqc_batch_norm_stats(
    model: TQC,
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """
    Clone the bias and running mean from the actor and critic networks and critic-target networks.
    :param model:
    :return: the bias and running mean from the actor and critic networks and critic-target networks
    """
    actor_batch_norm = model.actor.features_extractor.batch_norm
    actor_bias, actor_running_mean = clone_batch_norm_stats(actor_batch_norm)

    critic_batch_norm = model.critic.features_extractor.batch_norm
    critic_bias, critic_running_mean = clone_batch_norm_stats(critic_batch_norm)

    critic_target_batch_norm = model.critic_target.features_extractor.batch_norm
    critic_target_bias, critic_target_running_mean = clone_batch_norm_stats(critic_target_batch_norm)

    return (actor_bias, actor_running_mean, critic_bias, critic_running_mean, critic_target_bias, critic_target_running_mean)


def clone_on_policy_batch_norm(model: Union[MaskablePPO]) -> (th.Tensor, th.Tensor):
    return clone_batch_norm_stats(model.policy.features_extractor.batch_norm)


CLONE_HELPERS = {
    QRDQN: clone_qrdqn_batch_norm_stats,
    TQC: clone_tqc_batch_norm_stats,
    MaskablePPO: clone_on_policy_batch_norm,
    ARDQN: clone_ar_ddqn_batch_norm_stats,
}

#
# def test_ppo_mask_train_eval_mode():
#     env = InvalidActionEnvDiscrete(dim=20, n_invalid_actions=10)
#     model = MaskablePPO(
#         "MlpPolicy",
#         env,
#         policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
#         seed=1,
#     )
#
#     bias_before, running_mean_before = clone_on_policy_batch_norm(model)
#
#     model.learn(total_timesteps=200)
#
#     bias_after, running_mean_after = clone_on_policy_batch_norm(model)
#
#     assert ~th.isclose(bias_before, bias_after).all()
#     assert ~th.isclose(running_mean_before, running_mean_after).all()
#
#     batch_norm_stats_before = clone_on_policy_batch_norm(model)
#
#     observation, _ = env.reset()
#     action_masks = get_action_masks(env)
#     first_prediction, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
#     for _ in range(5):
#         prediction, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
#         np.testing.assert_allclose(first_prediction, prediction)
#
#     batch_norm_stats_after = clone_on_policy_batch_norm(model)
#
#     # No change in batch norm params
#     for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
#         assert th.isclose(param_before, param_after).all()
#
#
# def test_qrdqn_train_with_batch_norm():
#     model = QRDQN(
#         "MlpPolicy",
#         "CartPole-v1",
#         policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
#         learning_starts=0,
#         seed=1,
#         tau=0,  # do not clone the target
#     )
#
#     (
#         quantile_net_bias_before,
#         quantile_net_running_mean_before,
#         quantile_net_target_bias_before,
#         quantile_net_target_running_mean_before,
#     ) = clone_qrdqn_batch_norm_stats(model)
#
#     model.learn(total_timesteps=200)
#     # Force stats copy
#     model.target_update_interval = 1
#     model._on_step()
#
#     (
#         quantile_net_bias_after,
#         quantile_net_running_mean_after,
#         quantile_net_target_bias_after,
#         quantile_net_target_running_mean_after,
#     ) = clone_qrdqn_batch_norm_stats(model)
#
#     assert ~th.isclose(quantile_net_bias_before, quantile_net_bias_after).all()
#     # Running stat should be copied even when tau=0
#     assert th.isclose(quantile_net_running_mean_before, quantile_net_target_running_mean_before).all()
#
#     assert th.isclose(quantile_net_target_bias_before, quantile_net_target_bias_after).all()
#     # Running stat should be copied even when tau=0
#     assert th.isclose(quantile_net_running_mean_after, quantile_net_target_running_mean_after).all()
#
#
# def test_tqc_train_with_batch_norm():
#     model = TQC(
#         "MlpPolicy",
#         "Pendulum-v1",
#         policy_kwargs=dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor),
#         learning_starts=0,
#         tau=0,  # do not copy the target
#         seed=1,
#     )
#
#     (
#         actor_bias_before,
#         actor_running_mean_before,
#         critic_bias_before,
#         critic_running_mean_before,
#         critic_target_bias_before,
#         critic_target_running_mean_before,
#     ) = clone_tqc_batch_norm_stats(model)
#
#     model.learn(total_timesteps=200)
#     # Force stats copy
#     model.target_update_interval = 1
#     model._on_step()
#
#     (
#         actor_bias_after,
#         actor_running_mean_after,
#         critic_bias_after,
#         critic_running_mean_after,
#         critic_target_bias_after,
#         critic_target_running_mean_after,
#     ) = clone_tqc_batch_norm_stats(model)
#
#     assert ~th.isclose(actor_bias_before, actor_bias_after).all()
#     assert ~th.isclose(actor_running_mean_before, actor_running_mean_after).all()
#
#     assert ~th.isclose(critic_bias_before, critic_bias_after).all()
#     # Running stat should be copied even when tau=0
#     assert th.isclose(critic_running_mean_before, critic_target_running_mean_before).all()
#
#     assert th.isclose(critic_target_bias_before, critic_target_bias_after).all()
#     # Running stat should be copied even when tau=0
#     assert th.isclose(critic_running_mean_after, critic_target_running_mean_after).all()


# @pytest.mark.parametrize("model_class", [QRDQN, TQC, ArDQN])
@pytest.mark.parametrize("model_class", [ARDQN])
@pytest.mark.parametrize("ardqn_share", ["all", "none", "features_extractor"])
def test_offpolicy_collect_rollout_batch_norm(model_class, ardqn_share):
    if model_class in [QRDQN, ARDQN]:
        env_id = "CartPole-v1"
    else:
        env_id = "Pendulum-v1"

    clone_helper = CLONE_HELPERS[model_class]
    policy_kwargs = dict(net_arch=[16, 16], features_extractor_class=FlattenBatchNormDropoutExtractor)
    learning_starts = 10
    kwargs = {}
    if model_class in {ARDQN}:
        kwargs["initial_aspiration"] = 0.0
        policy_kwargs["shared_network"] = ardqn_share
    elif ardqn_share != "all":
        pytest.skip()

    model = model_class(
        "MlpPolicy",
        env_id,
        policy_kwargs=policy_kwargs,
        learning_starts=learning_starts,
        seed=1,
        gradient_steps=0,
        train_freq=1,
        **kwargs,
    )

    batch_norm_stats_before = clone_helper(model)

    model.learn(total_timesteps=100)

    batch_norm_stats_after = clone_helper(model)

    # No change in batch norm params
    for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
        assert th.isclose(param_before, param_after).all()


# @pytest.mark.parametrize("model_class", [QRDQN, TQC, ARDQN])
@pytest.mark.parametrize("model_class", [ARDQN])
@pytest.mark.parametrize("env_id", ["Pendulum-v1", "CartPole-v1"])
def test_predict_with_dropout_batch_norm(model_class, env_id):
    if env_id == "CartPole-v1":
        if model_class in [TQC]:
            return
    elif model_class in [QRDQN, ARDQN]:
        return

    model_kwargs = dict(seed=1)
    clone_helper = CLONE_HELPERS[model_class]
    policy_kwargs = dict(
        features_extractor_class=FlattenBatchNormDropoutExtractor,
        net_arch=[16, 16],
    )
    if model_class in [QRDQN, TQC, ARDQN]:
        model_kwargs["learning_starts"] = 0
    else:
        model_kwargs["n_steps"] = 64
    if model_class in [ARDQN]:
        model_kwargs["initial_aspiration"] = 10.0

    model = model_class("MlpPolicy", env_id, policy_kwargs=policy_kwargs, verbose=1, **model_kwargs)

    batch_norm_stats_before = clone_helper(model)

    env = model.get_env()
    observation = env.reset()
    first_prediction, _ = model.predict(observation, deterministic=True)
    for _ in range(5):
        prediction, _ = model.predict(observation, deterministic=True)
        np.testing.assert_allclose(first_prediction, prediction)

    batch_norm_stats_after = clone_helper(model)

    # No change in batch norm params
    for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
        assert th.isclose(param_before, param_after).all()


@pytest.mark.parametrize("env_id", ["CartPole-v1"])
@pytest.mark.parametrize("shared_network", ["all", "features_extractor", "none", "min_max"])
def test_ardqn_predict_with_dropout_batch_norm(env_id, shared_network):
    model_kwargs = dict(seed=1, learning_starts=0)
    clone_helper = CLONE_HELPERS[ARDQN]
    policy_kwargs = dict(
        features_extractor_class=FlattenBatchNormDropoutExtractor,
        net_arch=[16, 16],
        shared_network=shared_network,
    )
    model = ARDQN("MlpPolicy", env_id, initial_aspiration=10.0, policy_kwargs=policy_kwargs, verbose=1, **model_kwargs)

    batch_norm_stats_before = clone_helper(model)

    env = model.get_env()
    observation = env.reset()
    first_prediction, _ = model.predict(observation, deterministic=True)
    for _ in range(5):
        prediction, _ = model.predict(observation, deterministic=True)
        np.testing.assert_allclose(first_prediction, prediction)

    batch_norm_stats_after = clone_helper(model)

    # No change in batch norm params
    for param_before, param_after in zip(batch_norm_stats_before, batch_norm_stats_after):
        assert th.isclose(param_before, param_after).all()
