from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.policies import BaseModel, BasePolicy

from sb3_contrib.common.satisficing.utils import interpolate, ratio


class ARQPolicy(BasePolicy):

    device: th.device
    rho: float
    gamma: float

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        initial_aspiration: float,
        use_delta_predictor: bool,
        *,
        gamma,
        rho,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            **kwargs,
        )
        self.initial_aspiration = initial_aspiration
        self.use_delta_predictor = use_delta_predictor
        self.aspiration: np.ndarray = np.array(initial_aspiration)
        self.gamma = gamma
        self.rho = rho

    def _create_aliases(
        self,
        q_predictor: BaseModel,
        q_target_predictor: BaseModel,
        delta_qmin_predictor: BaseModel,
        delta_qmin_target_predictor: BaseModel,
        delta_qmax_predictor: BaseModel,
        delta_qmax_target_predictor: BaseModel,
    ) -> None:
        # We need to create aliases because the predictors are not available at init time
        # They are stored as lambda functions to avoid self.q_predictor to be counted in policy.parameters()
        self.q_predictor = lambda obs: q_predictor(obs)
        self.q_target_predictor = lambda obs: q_target_predictor(obs)
        self.delta_qmin_predictor = lambda obs: delta_qmin_predictor(obs)
        self.delta_qmin_target_predictor = lambda obs: delta_qmin_target_predictor(obs)
        self.delta_qmax_predictor = lambda obs: delta_qmax_predictor(obs)
        self.delta_qmax_target_predictor = lambda obs: delta_qmax_target_predictor(obs)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values_batch = self.q_predictor(obs)
        actions = th.zeros(len(obs), dtype=th.int)
        aspirations = th.as_tensor(self.aspiration, device=self.device).squeeze()
        # todo?: using a for loop may be crappy, if it's too slow, we could rewrite this using pytorch
        batch_size = len(list(obs.values())[0]) if isinstance(obs, dict) else len(obs)
        for i in range(batch_size):
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
                elif not lower.any():
                    # if all values are higher than aspiration, return the lowest value
                    actions[i] = q_values.argmin()
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
        return actions

    def rescale_aspiration(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        use_q_target: bool = True,
    ) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration as its return-to-go.

        :param obs: observations at time t
        :param actions: actions at time t
        :param rewards: rewards at time t
        :param next_obs: observations at time t+1
        :param use_q_target: whether to use the Q-value or the target Q-value
        """
        obs, next_obs = self.obs_to_tensor(obs)[0], self.obs_to_tensor(next_obs)[0]
        with th.no_grad():
            actions = th.as_tensor(actions, device=self.device, dtype=th.int64).unsqueeze(dim=1)
            q = th.gather(self.q_predictor(obs), dim=1, index=actions)
            qmin_predictor = self.delta_qmin_target_predictor if use_q_target else self.delta_qmin_predictor
            qmax_predictor = self.delta_qmax_target_predictor if use_q_target else self.delta_qmax_predictor
            q_min: th.Tensor = q - th.gather(qmin_predictor(obs), 1, actions)
            q_max = q + th.gather(qmax_predictor(obs), 1, actions)
            next_lambda = ratio(q_min, q, q_max).squeeze(dim=1)
            # If q_max == q_min, we arbitrary set lambda to 0.5 as this should not matter
            next_lambda[(q_max == q_min).squeeze(dim=1)] = 0.5
            next_q = self.q_target_predictor(next_obs) if use_q_target else self.q_predictor(next_obs)
            next_aspiration = interpolate(next_q.min(dim=1).values, next_lambda, next_q.max(dim=1).values).cpu().numpy()
            delta_hard = -rewards / self.gamma
            delta_soft = next_aspiration - q.cpu().squeeze(dim=1).numpy() / self.gamma
            self.aspiration = self.aspiration / self.gamma + interpolate(delta_hard, self.rho, delta_soft)

    def reset_aspiration(self, dones: Optional[np.ndarray] = None) -> None:
        """
        Reset the current aspiration to the initial one

        :param dones: if not None, reset only the aspiration that correspond to the done environments
        """
        if dones is None or self.aspiration.ndim == 0:
            self.aspiration = self.initial_aspiration
        else:
            self.aspiration[dones] = self.initial_aspiration

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                initial_aspiration=self.initial_aspiration,
                rho=self.rho,
                gamma=self.gamma,
            )
        )
        return data

    def q_values(self, obs: np.ndarray, actions: Optional[np.ndarray] = None) -> th.Tensor:
        if actions is None:
            return self.q_predictor(self.obs_to_tensor(obs)[0])
        else:
            t_actions = th.as_tensor(actions, device=self.device, dtype=th.long).unsqueeze(dim=1)
            return self.q_predictor(self.obs_to_tensor(obs)[0]).gather(1, t_actions).squeeze(dim=1)

    def lambda_ratio(self, obs: np.ndarray, aspiration: Union[float, np.ndarray]) -> th.Tensor:
        q = self.q_values(obs)
        q_min = q.min(dim=1).values
        q_max = q.max(dim=1).values
        lambdas = ratio(q_min, th.tensor(aspiration, device=self.device), q_max)
        lambdas[q_max == q_min] = 0.5  # If q_max == q_min, we set lambda to 0.5, this should not matter
        return lambdas
