from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor, FlattenExtractor, NatureCNN
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.dqn.policies import QNetwork
from torch import nn, relu

from sb3_contrib.common.satisficing.policies import ARQPolicy


class HydraNetwork(QNetwork):
    """
    Hydra network for AR-DQN. This QNetwork has three heads, one for the q-values, one for the delta_qmin and
    one for the delta_qmax.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            spaces.Discrete(action_space.n * 3),
            features_extractor,
            features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        self.real_action_space = action_space

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Predict the q-values.

        :param obs: Observation
        :return: A tuple containing the Q-values, delta_qmin and delta_qmax
        """
        out = super().forward(obs)
        q_values = out[:, : self.real_action_space.n]
        delta_qmin = out[:, self.real_action_space.n : self.real_action_space.n * 2]
        delta_qmax = out[:, self.real_action_space.n * 2 :]
        return q_values, relu(delta_qmin), relu(delta_qmax)

    def _make_head(self, index: int) -> HydraHead:
        return HydraHead(
            self,
            index,
            self.observation_space,
            self.real_action_space,
            self.features_extractor_class,
            self.features_extractor_kwargs,
            self.features_extractor,
            self.normalize_images,
            self.optimizer_class,
            self.optimizer_kwargs,
        )

    def create_heads(self) -> Tuple[HydraHead, HydraHead, HydraHead]:
        """
        Create the three heads of the Hydra network.

        :return: q_net, delta_qmin_net, delta_qmax_net
        """
        return self._make_head(0), self._make_head(1), self._make_head(2)


class HydraHead(BaseModel):
    def __init__(
        self,
        hydra_net: HydraNetwork,
        index: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.hydra_net = hydra_net
        self.index = index

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.hydra_net(obs)[self.index]


class ArDQNPolicy(ARQPolicy):
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

    q_net: BaseModel
    q_net_target: BaseModel
    delta_qmax_net: BaseModel
    qmax_net_target: BaseModel
    delta_qmin_net: BaseModel
    qmin_net_target: BaseModel

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        initial_aspiration: float,
        gamma: float,
        rho: float,
        shared_network: Literal["features_extractor", "all", "none"] = "none",
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
            initial_aspiration,
            gamma=gamma,
            rho=rho,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.shared_network = shared_network
        self._build(lr_schedule, shared_network)

    def _build(self, lr_schedule: Schedule, shared_network) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        if shared_network == "none":
            self.q_net, self.q_net_target = self.make_q_nets()
            self.delta_qmin_net, self.delta_qmin_net_target = self.make_delta_q_nets()
            self.delta_qmax_net, self.delta_qmax_net_target = self.make_delta_q_nets()
        elif shared_network == "all":
            net_args = self._update_features_extractor(self.net_args, features_extractor=None)
            self.hydra_net = HydraNetwork(**net_args).to(self.device)
            self.q_net, self.delta_qmin_net, self.delta_qmax_net = self.hydra_net.create_heads()
            self.hydra_net_target = HydraNetwork(**net_args).to(self.device)
            self.hydra_net_target.load_state_dict(self.hydra_net.state_dict())
            self.hydra_net_target.set_training_mode(False)
            self.q_net_target, self.delta_qmin_net_target, self.delta_qmax_net_target = self.hydra_net_target.create_heads()
        elif shared_network == "features_extractor":
            self.q_net, self.q_net_target = self.make_q_nets()
            self.delta_qmin_net, self.delta_qmin_net_target = self.make_delta_q_nets()
            self.delta_qmax_net, self.delta_qmax_net_target = self.make_delta_q_nets()
            features_extractor = self.q_net.features_extractor
            self.delta_qmin_net.features_extractor = features_extractor
            self.delta_qmax_net.features_extractor = features_extractor
            target_features_extractor = self.q_net_target.features_extractor
            self.delta_qmin_net_target.features_extractor = target_features_extractor
            self.delta_qmax_net_target.features_extractor = target_features_extractor
        else:
            raise NotImplementedError(
                f"Unknown shared_network value: {shared_network}\nPlease use one of: 'none', 'all', 'features_extractor'"
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        super()._create_aliases(
            self.q_net,
            self.q_net_target,
            self.delta_qmin_net,
            self.delta_qmin_net_target,
            self.delta_qmax_net,
            self.delta_qmax_net_target,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

    def make_q_nets(self) -> Tuple[QNetwork, QNetwork]:
        # Make sure we always have separate networks for features extractors etc
        q_net = self.make_q_net()
        q_net_target = self.make_q_net()
        q_net_target.load_state_dict(q_net.state_dict())
        q_net_target.set_training_mode(False)
        return q_net, q_net_target

    def make_delta_q_nets(self) -> Tuple[QNetwork, QNetwork]:
        delta_q_net = self.make_q_net()
        delta_q_net.q_net = nn.Sequential(delta_q_net.q_net, nn.ReLU())
        delta_q_net_target = self.make_q_net()
        delta_q_net_target.q_net = nn.Sequential(delta_q_net_target.q_net, nn.ReLU())
        delta_q_net_target.load_state_dict(delta_q_net.state_dict())
        delta_q_net_target.set_training_mode(False)
        return delta_q_net, delta_q_net_target

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        if self.shared_network in {"none", "features_extractor"}:
            self.q_net.set_training_mode(mode)
            self.delta_qmax_net.set_training_mode(mode)
            self.delta_qmin_net.set_training_mode(mode)
        elif self.shared_network == "all":
            self.hydra_net.set_training_mode(mode)
        self.training = mode

    def update_target_nets(self, tau: float) -> None:
        if self.shared_network == "none":
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), tau)
            polyak_update(self.delta_qmax_net.parameters(), self.delta_qmax_net_target.parameters(), tau)
            polyak_update(self.delta_qmin_net.parameters(), self.delta_qmin_net_target.parameters(), tau)
        elif self.shared_network == "all":
            polyak_update(self.hydra_net.parameters(), self.hydra_net_target.parameters(), tau)
        elif self.shared_network == "features_extractor":
            polyak_update(self.q_net.q_net.parameters(), self.q_net_target.q_net.parameters(), tau)
            polyak_update(self.delta_qmax_net.q_net.parameters(), self.delta_qmax_net_target.q_net.parameters(), tau)
            polyak_update(self.delta_qmin_net.q_net.parameters(), self.delta_qmin_net_target.q_net.parameters(), tau)
            # Update the features extractor separately to avoid updating it three times
            polyak_update(self.q_net.features_extractor.parameters(), self.q_net_target.features_extractor.parameters(), tau)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                shared_network=self.shared_network,
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data


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
        initial_aspiration: float,
        gamma: float,
        rho: float,
        shared_network: Literal["features_extractor", "all", "none"] = "none",
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
            gamma,
            rho,
            shared_network=shared_network,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
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
        gamma: float,
        rho: float,
        shared_network: Literal["features_extractor", "all", "none"] = "none",
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
            gamma,
            rho,
            shared_network=shared_network,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
