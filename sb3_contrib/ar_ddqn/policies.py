from typing import Any, Dict, Iterator, List, Optional, Type

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn
from torch.nn import Parameter



class ArDDQNPolicy(DQNPolicy):
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
    qmax_net: QNetwork
    qmax_net_target: QNetwork
    qmin_net: QNetwork
    qmin_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        init_aspiration: float,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.aspiration = init_aspiration
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.qmax_net = self.make_q_net()
        self.qmin_net = self.make_q_net()
        # Add a ReLU to the last layer of qmax_net and qmin_net because we want to have positive values
        self.qmax_net.q_net = nn.Sequential(self.qmax_net.q_net, nn.ReLU())
        self.qmin_net.q_net = nn.Sequential(self.qmin_net.q_net, nn.ReLU())
        # Super methode will create the optimizer, q_net and q_net_target
        super()._build()

    # def make_q_nets(self, net_args) -> (QNetwork, QNetwork):
    #     # todo: remove if no target are needed for qmin and qmax
    #     """
    #     Create the network and the target network.
    #     The target network is a copy of the network with the initial weights and it's not on training mode.
    #
    #     :return: A pair containing the network and the target network
    #     """
    #     # Make sure we always have separate networks for features extractors etc
    #     net = QNetwork(**net_args).to(self.device)
    #     target_net = QNetwork(**net_args).to(self.device)
    #     target_net.load_state_dict(net.state_dict())
    #     target_net.set_training_mode(False)
    #     return net, target_net

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.q_net(obs)
        exact = (q_values == self.aspiration).nonzero()
        if len(exact) > 0:
            return np.random.sample(exact[0])
        else:
            higher = q_values > self.aspiration
            lower = q_values <= self.aspiration
            if not higher.any():
                # if all values are lower than aspiration, return the highest value
                return q_values.argmax()
            elif not lower.any():
                # if all values are higher than aspiration, return the lowest value
                return q_values.argmin()
            else:
                a_minus = q_values[lower].argmax()
                a_plus = q_values[higher].argmin()
                p = (self.aspiration - q_values[a_minus]) / (q_values[a_plus] - q_values[a_minus])
                # Else, with probability p return a+
                if np.random.rand() <= p:
                    return a_plus
                else:
                    return a_minus

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.qmax_net.set_training_mode(mode)
        self.qmin_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = ArDDQNPolicy
# todo : check but I think we can use the DQN one
#
# class CnnPolicyAr(ArDDQNPolicy):
#     """
#     Policy class for DQN when using images as input.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param features_extractor_class: Features extractor to use.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Discrete,
#         lr_schedule: Schedule,
#         net_arch: Optional[List[int]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )
#
#
# class MultiInputPolicyAr(ArDDQNPolicy):
#     """
#     Policy class for DQN when using dict observations as input.
#
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param features_extractor_class: Features extractor to use.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """
#
#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space: spaces.Discrete,
#         lr_schedule: Schedule,
#         net_arch: Optional[List[int]] = None,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             features_extractor_class,
#             features_extractor_kwargs,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )
