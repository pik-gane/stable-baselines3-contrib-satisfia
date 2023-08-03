from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from torch import nn

from sb3_contrib.common.satisficing.utils import interpolate, ratio


class LRADQNPolicy(DQNPolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: QNetwork
    q_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *,
        local_relative_aspiration: float,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.local_relative_aspiration = local_relative_aspiration

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values_batch = self.q_net(obs)
        local_aspirations = interpolate(
            q_values_batch.min(dim=1).values, self.local_relative_aspiration, q_values_batch.max(dim=1).values
        )
        # todo?: Maybe we can avoid using a for loop, but I'm not sure if it's worth it
        batch_size = len(list(obs.values())[0]) if isinstance(obs, dict) else len(obs)
        actions = th.zeros(batch_size, dtype=th.long, device=self.device)
        for i in range(batch_size):
            q_values: th.Tensor = q_values_batch[i]
            local_aspiration = local_aspirations[i]
            exact = (q_values == local_aspiration).nonzero()
            if len(exact) > 0:
                if not deterministic:
                    # Choose randomly among actions that satisfy the aspiration
                    index = np.random.randint(0, len(exact[0]))
                    actions[i] = exact[0][index]
                else:
                    actions[i] = exact[0].min()
            else:
                higher = q_values > local_aspiration
                lower = q_values < local_aspiration
                q_values_for_max = q_values.clone()
                q_values_for_max[lower] = th.inf
                q_values_for_min = q_values.clone()
                q_values_for_min[higher] = -th.inf
                a_minus = q_values_for_min.argmax()
                a_plus = q_values_for_max.argmin()
                p = ratio(q_values[a_minus], local_aspiration, q_values[a_plus])
                # Else, with probability p return a+
                if (not deterministic and np.random.rand() <= p) or (p > 0.5 and deterministic):
                    actions[i] = a_plus
                else:
                    actions[i] = a_minus
        return actions

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(dict(local_relative_aspiration=self.local_relative_aspiration))
        return data


MlpPolicy = LRADQNPolicy


class CnnPolicy(LRADQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        local_relative_aspiration: float,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            local_relative_aspiration=local_relative_aspiration,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


class MultiInputPolicy(LRADQNPolicy):
    """
    Policy class for DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *,
        local_relative_aspiration: float,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            local_relative_aspiration=local_relative_aspiration,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
