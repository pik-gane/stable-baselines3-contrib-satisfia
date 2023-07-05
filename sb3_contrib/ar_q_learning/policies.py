import warnings
from typing import Optional, Dict, Any, Union

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
        # self.q_table += th.randn_like(self.q_table) * 0.5  # todo Suggest: Maybe this break the symmetry

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

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values_batch = self.q_table(obs)
        actions = th.zeros(len(obs), dtype=th.int)
        aspirations = th.as_tensor(self.aspiration, device=self.device).squeeze()
        shortfall = th.zeros(len(obs), dtype=th.float)
        excess = th.zeros(len(obs), dtype=th.float)
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
                    shortfall[i] = aspiration - q_values[actions[i]]
                elif not lower.any():
                    # if all values are higher than aspiration, return the lowest value
                    actions[i] = q_values.argmin()
                    excess[i] = q_values[actions[i]] - aspiration
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
        self.shortfall = shortfall  # Jobst: is this a safe way of passing the shortfall and excess to the rescale method?
        self.excess = excess
        return actions

    def rescale_aspiration(
        self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray, dones: Optional[np.ndarray] = None
    ) -> None:
        # todo remove dones
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
            q = th.gather(
                self.q_table(obs), dim=1, index=actions
            )
            q_min = q - th.gather(self.delta_qmin_table(obs), 1, actions)
            q_max = q + th.gather(self.delta_qmax_table(obs), 1, actions)
            # We need to use nan_to_num here, just in case delta qmin and qmax are 0. The value 0.5 is arbitrarily
            #   chosen as in theory it shouldn't matter.
            lambda_t1 = ratio(q_min, q, q_max).squeeze(dim=1).nan_to_num(nan=0.5)
            # squeeze: n * 1 -> n
            next_q = self.q_table(next_obs)
            self.aspiration = (
                interpolate(next_q.min(dim=1).values, lambda_t1, next_q.max(dim=1).values).cpu().numpy()
            ) + self.shortfall - self.excess
        if (
            dones is not None and (q_min == q_max)[1 - dones].any()
        ):  # todo remove this once we are sure the .nan_to_num is not a problem
            warnings.warn(
                "q_min and q_max are equal, this is weird. Happened for aspiration {}".format(self.initial_aspiration)
            )

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
