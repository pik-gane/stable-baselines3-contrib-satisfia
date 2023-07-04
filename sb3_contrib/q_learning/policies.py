from typing import Union, Optional

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.policies import BasePolicy
from torch import nn


class QTable(BasePolicy):  # Jobst: Why is this a policy? It doesn't predict an action, but a Q-value. Maybe it should be a subclass of BaseModel instead?
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

    def forward(self, obs: Union[th.Tensor, np.ndarray],
                action: Optional[Union[th.Tensor, np.ndarray]] = None) -> th.Tensor:
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

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        raise NotImplementedError("A Q table can't predict")

    def update_table(
        self, obs: Union[np.ndarray, th.Tensor], 
        actions: Union[np.ndarray, th.Tensor],  # Jobst: Why is this plural? Shouldn't it be action?
        target: th.Tensor, lr: float
    ) -> None:
        """
        Update the Q-Table.

        :param obs: Observation
        :param actions: Actions
        :param target: Target Q-Values
        :param lr: Learning rate
        """
        self.q_table[obs, actions] += lr * (target - self.q_table[obs, actions])


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
        return self.q_table(obs).argmax(dim=1).reshape(-1)  # Jobst: is q_table(obs) calling the forward method of QTable? If so, doesn't that return a 1d tensor, indexed on action alone? If so, why do you have dim=1 instead of dim=0 or no dim at all?
