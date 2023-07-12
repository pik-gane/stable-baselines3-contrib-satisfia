from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.policies import BasePolicy

from sb3_contrib.common.satisficing.policies import ARQPolicy
from sb3_contrib.common.satisficing.utils import interpolate, ratio
from sb3_contrib.q_learning.policies import QTable


class DeltaQTable(QTable):
    def __init__(
        self,
        observation_space: spaces.Discrete,
        action_space: spaces.Discrete,
    ) -> None:
        super().__init__(observation_space, action_space)
        self.q_table[...] = 1.0 + th.randn(self.q_table.shape) * 0.05

    def forward(self, obs: Union[th.Tensor, np.ndarray], action: Optional[Union[th.Tensor, np.ndarray]] = None) -> th.Tensor:
        return th.nn.functional.relu(super().forward(obs, action))


class ARQLearningPolicy(ARQPolicy):
    """
    Policy object that implements a Q-Table-based absolute aspiration-based satisficing policy with Aspiration Rescaling.

    :param observation_space: Observation space
    :param action_space: Action space
    """

    q_table: QTable
    delta_qmax_table: DeltaQTable
    delta_qmin_table: DeltaQTable

    def __init__(
        self,
        observation_space: spaces.Discrete,
        action_space: spaces.Discrete,
        initial_aspiration: float,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            initial_aspiration,
        )
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.q_table = QTable(self.observation_space, self.action_space).to(self.device)
        self.delta_qmax_table = DeltaQTable(self.observation_space, self.action_space).to(self.device)
        self.delta_qmin_table = DeltaQTable(self.observation_space, self.action_space).to(self.device)
        super()._create_aliases(self.q_table, self.q_table, self.delta_qmin_table, self.delta_qmax_table)
