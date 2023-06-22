import warnings
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

from sb3_contrib.ar_dqn.utils import interpolate, ratio


class ArDQNPolicy(DQNPolicy):
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
    delta_qmax_net: QNetwork
    qmax_net_target: QNetwork
    delta_qmin_net: QNetwork
    qmin_net_target: QNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        initial_aspiration: float,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.aspiration = initial_aspiration
        self.initial_aspiration = initial_aspiration
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
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.delta_qmax_net = self.make_q_net()
        self.delta_qmin_net = self.make_q_net()
        # Add a ReLU to the last layer of delta_qmax_net and delta_qmin_net because we want to have positive values
        self.delta_qmax_net.q_net = nn.Sequential(self.delta_qmax_net.q_net, nn.ReLU())
        self.delta_qmin_net.q_net = nn.Sequential(self.delta_qmin_net.q_net, nn.ReLU())
        # Super methode will create the optimizer, q_net and q_net_target put set q_net_target into eval mode
        super()._build(lr_schedule)

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

    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        If deterministic is false, will return the action which Q-value is closest to the aspiration.
        """
        # todo?: using a for loop is crappy, if it's too slow, we could rewrite this using pytorch
        q_values_batch = self.q_net(obs)
        actions = th.zeros(len(obs), dtype=th.int)
        aspirations = th.as_tensor(self.aspiration, device=self.device)
        for i in range(len(obs)):
            q_values = q_values_batch[i]
            if aspirations.dim() > 0:
                aspiration = aspirations[i]
            else:
                aspiration = aspirations
            exact = (q_values == aspiration).nonzero()
            if len(exact) > 0:
                index = np.random.randint(0, len(exact[0]))
                actions[i] = exact[0][index]
            else:
                higher = q_values > aspiration
                lower = q_values <= aspiration
                if not higher.any():
                    # if all values are lower than aspiration, return the highest value
                    actions[i] = q_values.argmax()
                elif not lower.any():
                    # if all values are higher than aspiration, return the lowest value
                    actions[i] = q_values.argmin()
                else:
                    a_minus = q_values[lower].argmax()
                    a_plus = q_values[higher].argmin()
                    p = ratio(q_values[a_minus], aspiration, q_values[a_plus])
                    # Else, with probability p return a+
                    if np.random.rand() <= p or (p > 0.5 and deterministic):
                        actions[i] = a_plus
                    else:
                        actions[i] = a_minus
        return actions

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.delta_qmax_net.set_training_mode(mode)
        self.delta_qmin_net.set_training_mode(mode)
        self.training = mode

    def rescale_aspiration(self, obs_t: th.Tensor, a_t: np.ndarray, obs_t1: th.Tensor) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration.

        :param obs_t: observation at time t
        :param a_t: action at time t
        :param obs_t1: observation at time t+1
        """
        with th.no_grad():
            a_t = th.as_tensor(a_t, device=self.device, dtype=th.int64).unsqueeze(dim=1)
            q = th.gather(self.q_net(obs_t), dim=1, index=a_t)
            q_min = q - th.gather(self.delta_qmin_net(obs_t), 1, a_t)
            q_max = q + th.gather(self.delta_qmax_net(obs_t), 1, a_t)
            # We need to use nan_to_num here, just in case delta qmin and qmax are 0. The value 0.5 is arbitrarily chosen
            #   as in theory it shouldn't matter.
            lambda_t1 = ratio(q_min, q, q_max).squeeze(dim=1).nan_to_num(nan=0.5)
            q = self.q_net_target(obs_t1)
            self.aspiration = interpolate(q.min(dim=1).values, lambda_t1, q.max(dim=1).values).cpu().numpy()

    def reset_aspiration(self) -> None:
        """
        Reset the current aspiration to the initial one
        """
        self.aspiration = self.initial_aspiration


MlpPolicy = ArDQNPolicy


class CnnPolicy(ArDQNPolicy):
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
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        *,
        initial_aspiration: float,
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
            initial_aspiration,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class MultiInputPolicy(ArDQNPolicy):
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
        initial_aspiration: float,
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
            lr_schedule,
            initial_aspiration,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
