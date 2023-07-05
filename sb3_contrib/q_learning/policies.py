from typing import Union, Optional

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.policies import BasePolicy, BaseModel
from torch import nn


class QTable(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Discrete,
        action_space: spaces.Discrete,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
        )
        self.q_table = nn.Parameter(th.zeros(observation_space.n, action_space.n), requires_grad=False)

    def forward(self, obs: Union[th.Tensor, np.ndarray], action: Optional[Union[th.Tensor, np.ndarray]] = None) -> th.Tensor:
        """
        Predict the q-values. Returns the Q values for all actions if `action` is None

        :param obs: Observation
        :param action: Action | None
        :return: The estimated Q-Value for the chosen action or for each action.
        """
        if action is None:
            return self.q_table[obs]
        else:
            return self.q_table[obs, action]

    def update_table(
        self, obs: Union[np.ndarray, th.Tensor],
        actions: Union[np.ndarray, th.Tensor],
        target: th.Tensor, lr: float
    ) -> None:
        """
        Update the Q-Table.

        :param obs: Observation
        :param actions: Actions
        :param target: Target Q-Values
        :param lr: Learning rate
        """
        self.q_table[obs, actions] += lr * (target - self.q_table[obs, actions]).mean()


class QLearningPolicy(BasePolicy):
    """
    Policy object that implements a Q-Table-based greedy (argmax) policy.

    :param observation_space: Observation space
    :param action_space: Action space
    """

    q_table: QTable

    def __init__(
        self,
        observation_space: spaces.Discrete,
        action_space: spaces.Discrete,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
        )
        self._build()

    def _build(self) -> None:
        self.q_table = QTable(self.observation_space, self.action_space)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Obs : n, self.q_table(obs): n*A argmax-> n*1 reshape->n
        return self.q_table(obs).argmax(dim=1).reshape(-1)
