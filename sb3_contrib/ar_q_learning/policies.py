import warnings
from typing import Optional, Dict, Any, Union, Tuple

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.policies import BasePolicy

from sb3_contrib.common.satisficing.utils import ratio, interpolate
from sb3_contrib.q_learning.policies import QTable


class DeltaQTable(QTable):
    def __init__(
        self,
        observation_space: spaces.Discrete,
        action_space: spaces.Discrete,
    ) -> None:
        super().__init__(observation_space, action_space)
        self.q_table[...] = 1.0

    def forward(self, obs: Union[th.Tensor, np.ndarray], action: Optional[Union[th.Tensor, np.ndarray]] = None) -> th.Tensor:
        return th.nn.functional.relu(super().forward(obs, action))


class ARQLearningPolicy(BasePolicy):
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
        )
        self.initial_aspiration = initial_aspiration
        self.aspiration = initial_aspiration
        self._build()

    def _build(self) -> None:
        self.q_table = QTable(self.observation_space, self.action_space).to(self.device)
        self.delta_qmax_table = DeltaQTable(self.observation_space, self.action_space).to(self.device)
        self.delta_qmin_table = DeltaQTable(self.observation_space, self.action_space).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        q_values_batch = self.q_table(obs)
        actions = th.zeros(len(obs), dtype=th.int)
        aspirations = th.as_tensor(self.aspiration, device=self.device).squeeze()
        aspiration_diffs = th.zeros(len(obs), dtype=th.float, device=self.device)
        for i in range(len(obs)):
            q_values: th.Tensor = q_values_batch[i]
            if aspirations.dim() > 0:
                aspiration = aspirations[i]
            else:
                aspiration = aspirations
            exact = (q_values == aspiration).nonzero()
            if len(exact) > 0:
                if not deterministic:
                    # Choose randomly among actions that satisfy the aspiration
                    index = np.random.randint(0, len(exact[0]))
                    actions[i] = exact[0][index]
                else:
                    actions[i] = exact[0].min()
            else:
                higher = q_values > aspiration
                lower = q_values < aspiration
                if not higher.any():
                    # if all values are lower than aspiration, return the highest value
                    actions[i] = q_values.argmax()
                    aspiration_diffs[i] = aspiration - q_values[actions[i]]
                elif not lower.any():
                    # if all values are higher than aspiration, return the lowest value
                    actions[i] = q_values.argmin()
                    aspiration_diffs[i] = aspiration - q_values[actions[i]]
                else:
                    q_values_for_max = q_values.clone()
                    q_values_for_max[lower] = th.inf
                    q_values_for_min = q_values.clone()
                    q_values_for_min[higher] = -th.inf
                    a_minus = q_values_for_min.argmax()
                    a_plus = q_values_for_max.argmin()
                    p = ratio(q_values[a_minus], aspiration, q_values[a_plus])
                    # Else, with probability p return a+
                    if (not deterministic and np.random.rand() <= p) or (p > 0.5 and deterministic):
                        actions[i] = a_plus
                    else:
                        actions[i] = a_minus
        return actions, aspiration_diffs

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: None (only used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the aspiration difference
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, aspiration_diff = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        aspiration_diff = aspiration_diff.cpu().numpy()

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, aspiration_diff

    def rescale_aspiration(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        next_obs: np.ndarray,
        aspiration_diffs: Optional[np.ndarray] = None,
    ) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration as its return-to-go.

        :param obs: observation at time t
        :param actions: action at time t
        :param next_obs: observation at time t+1
        """
        with th.no_grad():
            actions = th.as_tensor(actions, device=self.device, dtype=th.int64).unsqueeze(dim=1)
            # self.q_table(obs) : n * A, actions : n * 1 -> q : n * 1
            q = th.gather(self.q_table(obs), dim=1, index=actions)
            q_min = q - th.gather(self.delta_qmin_table(obs), 1, actions)
            q_max = q + th.gather(self.delta_qmax_table(obs), 1, actions)
            # We need to use nan_to_num here, just in case delta qmin and qmax are 0. The value 0.5 is arbitrarily
            #   chosen as in theory it shouldn't matter.
            lambda_t1 = ratio(q_min, q, q_max).squeeze(dim=1).nan_to_num(nan=0.5)
            # squeeze: n * 1 -> n
            next_q = self.q_table(next_obs)
            self.aspiration = interpolate(next_q.min(dim=1).values, lambda_t1, next_q.max(dim=1).values).cpu().numpy()
            if aspiration_diffs is not None:
                self.aspiration += aspiration_diffs

    def reset_aspiration(self, dones: Optional[np.ndarray] = None) -> None:
        """
        Reset the current aspiration to the initial one

        :param dones: if not None, reset only the aspiration that correspond to the done environments
        """
        if dones is None:
            self.aspiration = self.initial_aspiration
        else:
            self.aspiration[dones] = self.initial_aspiration

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                initial_aspiration=self.initial_aspiration,
            )
        )
        return data

    def q_values(self, obs: np.ndarray):
        return self.q_table(obs)

    def lambda_ratio(self, obs: np.ndarray, aspiration: Union[float, np.ndarray]) -> th.Tensor:
        q = self.q_values(obs)
        q_min = q.min(dim=1).values
        q_max = q.max(dim=1).values
        lambdas = ratio(q_min, th.tensor(aspiration, device=self.device), q_max)
        lambdas[q_max == q_min] = 0.5  # If q_max == q_min, we set lambda to 0.5, this should not matter
        return lambdas
