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
        use_delta_predictors: bool,
        *,
        gamma,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            **kwargs,
        )
        self.use_delta_predictors = use_delta_predictors
        self.initial_aspiration = initial_aspiration
        self.aspiration: np.ndarray = np.array(initial_aspiration)
        self.gamma = gamma

    def _create_aliases(
        self,
        q_predictor: BaseModel,
        q_target_predictor: BaseModel,
        qmin_predictor: BaseModel,
        qmin_target_predictor: BaseModel,
        qmax_predictor: BaseModel,
        qmax_target_predictor: BaseModel,
    ) -> None:
        # We need to create aliases because the predictors are not available at init time
        # They are stored as lambda functions to avoid self.q_predictor,... to be counted in policy.parameters()
        self.q_predictor = lambda obs: q_predictor(obs)
        self.q_target_predictor = lambda obs: q_target_predictor(obs)
        self.delta_qmin_predictor = (lambda obs: qmin_predictor(obs)) if self.use_delta_predictors else None
        self.delta_qmin_target_predictor = (lambda obs: qmin_target_predictor(obs)) if self.use_delta_predictors else None
        self.delta_qmax_predictor = (lambda obs: qmax_predictor(obs)) if self.use_delta_predictors else None
        self.delta_qmax_target_predictor = (lambda obs: qmax_target_predictor(obs)) if self.use_delta_predictors else None
        self.qmin_predictor = (lambda obs: qmin_predictor(obs)) if not self.use_delta_predictors else None
        self.qmin_target_predictor = (lambda obs: qmin_target_predictor(obs)) if not self.use_delta_predictors else None
        self.qmax_predictor = (lambda obs: qmax_predictor(obs)) if not self.use_delta_predictors else None
        self.qmax_target_predictor = (lambda obs: qmax_target_predictor(obs)) if not self.use_delta_predictors else None

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

    def propagate_aspiration(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration as its return-to-go.

        :param obs: observations at time t
        :param actions: actions at time t
        :param rewards: rewards at time t
        :param next_obs: observations at time t+1
        """
        obs, next_obs = self.obs_to_tensor(obs)[0], self.obs_to_tensor(next_obs)[0]
        with th.no_grad():
            actions = th.as_tensor(actions, device=self.device, dtype=th.int64).unsqueeze(dim=1)
            q = self.q_values(obs, actions=actions)
            q_min = self.qmin_values(obs, actions=actions)
            q_max = self.qmax_values(obs, actions=actions)
            next_lambda = ratio(q_min, q, q_max)
            # If q_max == q_min, we arbitrary set lambda to 0.5 as this should not matter
            next_lambda[(q_max == q_min)] = 0.5
            next_qs = self.q_predictor(next_obs)
            next_q = interpolate(next_qs.min(dim=1).values, next_lambda, next_qs.max(dim=1).values).cpu().numpy()
            delta_hard = -rewards / self.gamma
            delta_soft = next_q - q.cpu().numpy() / self.gamma
            self.aspiration = self.aspiration / self.gamma + interpolate(delta_hard, self.rho, delta_soft)

            # Check that in the end, the expected value of reward + gamma * aspiration(new) equals aspiration(old):
            # Equivalently, we want that
            # 0 = E(reward/gamma + aspiration(new) - self.aspiration(old)/gamma)
            #   = E(reward/gamma + interpolate(delta_hard, rho, delta_soft)
            #   = interpolate(E(reward/gamma + delta_hard), rho, E(reward/gamma + delta_soft))
            #   = interpolate(0, rho, E(reward/gamma + next_q - q/gamma))
            #   = rho * (E(reward/gamma) + interpolate(E(next_qs.min), ratio(q_min, q, q_max), E(next_qs.max)) - q/gamma)
            # Now we know that after the networks have been learned properly, we should have
            #   q_min ~ E(reward) + gamma * E(next_qs.min),
            #   q_max ~ E(reward) + gamma * E(next_qs.max),
            #   q ~ E(reward) + gamma * E(next_q),
            # so that
            #   next_lambda ~ ratio(q_min, q, q_max) ~ ratio(E(next_qs.min), E(next_q), E(next_qs.max))
            # hence we can rewrite the above as
            #   0 ~ rho * (E(reward/gamma) + interpolate(E(next_qs.min), next_lambda, E(next_qs.max)) - q/gamma)
            #     ~ rho * (E(reward/gamma) + E(next_q) - q/gamma)
            #     = 0, QED.

    def reset_aspiration(self, dones: Optional[np.ndarray] = None) -> None:
        """
        Reset the current aspiration to the initial one

        :param dones: if not None, reset only the aspiration that correspond to the done environments
        """
        if dones is None or self.aspiration.ndim == 0:
            self.aspiration = np.array(self.initial_aspiration)
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

    def get_q_values(self, obs: np.ndarray, actions: Optional[np.ndarray] = None) -> th.Tensor:
        """
        Get the Q values for the given observations and actions. If actions is None, return the Q values for all
        actions, otherwise return the Q values for the given actions only.

        :param obs: the observations
        :param actions: the actions
        """
        if actions is None:
            return self.q_predictor(self.obs_to_tensor(obs)[0])
        else:
            t_actions = th.as_tensor(actions, device=self.device, dtype=th.long).unsqueeze(dim=1)
            return self.q_predictor(self.obs_to_tensor(obs)[0]).gather(1, t_actions).squeeze(dim=1)

    def qmin_values(self, obs: th.Tensor, use_target: bool = False, actions: Optional[th.Tensor] = None) -> th.Tensor:
        if self.use_delta_predictors:
            if use_target:
                qmin = self.q_target_predictor(obs) - self.delta_qmin_target_predictor(obs)
            else:
                qmin = self.q_predictor(obs) - self.delta_qmin_predictor(obs)
        else:
            if use_target:
                qmin = self.qmin_target_predictor(obs)
            else:
                qmin = self.qmin_predictor(obs)
        if actions is None:
            return qmin
        else:
            return qmin.gather(1, actions).squeeze(dim=1)

    def qmax_values(self, obs: th.Tensor, use_target: bool = False, actions: Optional[th.Tensor] = None) -> th.Tensor:
        if self.use_delta_predictors:
            if use_target:
                qmax = self.q_target_predictor(obs) + self.delta_qmax_target_predictor(obs)
            else:
                qmax = self.q_predictor(obs) + self.delta_qmax_predictor(obs)
        else:
            if use_target:
                qmax = self.qmax_target_predictor(obs)
            else:
                qmax = self.qmax_predictor(obs)
        if actions is None:
            return qmax
        else:
            return qmax.gather(1, actions).squeeze(dim=1)

    def q_values(self, obs: th.Tensor, use_target: bool = False, actions: Optional[th.Tensor] = None) -> th.Tensor:
        if use_target:
            q = self.q_target_predictor(obs)
        else:
            q = self.q_predictor(obs)
        if actions is None:
            return q
        else:
            return q.gather(1, actions).squeeze(dim=1)

    def lambda_ratio(self, obs: np.ndarray, aspirations: Union[float, np.ndarray]) -> th.Tensor:
        """
        Get the lambda ratio for the given observations and aspiration. The lambda ratio is clamped between 0 and 1.
        Note: If the Q values are all equal, we set lambda to 0.5, (as this should not matter)

        :param obs: the observations
        :param aspirations: the aspirations

        :return: the clamped lambda ratio between 0 and 1
        """
        q = self.get_q_values(obs)
        q_min = q.min(dim=1).values
        q_max = q.max(dim=1).values
        lambdas = ratio(q_min, th.tensor(aspirations, device=self.device), q_max)
        lambdas[q_max == q_min] = 0.5  # If q_max == q_min, we set lambda to 0.5, this should not matter
        return lambdas.clamp(min=0, max=1)
