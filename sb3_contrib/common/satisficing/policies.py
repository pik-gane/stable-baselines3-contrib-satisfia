from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.policies import BaseModel, BasePolicy

from sb3_contrib.common.satisficing.utils import interpolate, ratio


class ARQPolicy(BasePolicy):

    device: th.device

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        initial_aspiration: float,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            **kwargs,
        )
        self.q_predictor = None
        self.q_target_predictor = None
        self.delta_qmin_predictor = None
        self.delta_qmax_predictor = None
        self.initial_aspiration = initial_aspiration
        self.aspiration: Union[float, np.ndarray] = initial_aspiration

    def _create_aliases(
        self,
        q_predictor: BaseModel,
        q_target_predictor: BaseModel,
        delta_qmin_predictor: BaseModel,
        delta_qmax_predictor: BaseModel,
    ) -> None:
        self.q_predictor = q_predictor
        self.q_target_predictor = q_target_predictor
        self.delta_qmin_predictor = delta_qmin_predictor
        self.delta_qmax_predictor = delta_qmax_predictor

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        q_values_batch = self.q_predictor(obs)
        actions = th.zeros(len(obs), dtype=th.int)
        aspirations = th.as_tensor(self.aspiration, device=self.device).squeeze()
        aspiration_diffs = th.zeros(len(obs), dtype=th.float, device=self.device)
        # todo?: using a for loop may be crappy, if it's too slow, we could rewrite this using pytorch
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
        _state: Optional[Tuple[np.ndarray, ...]] = None,
        _episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        use_q_target: bool = True,
    ) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration as its return-to-go.

        :param obs: observation at time t
        :param actions: action at time t
        :param next_obs: observation at time t+1
        :param aspiration_diffs: difference between the aspiration and the
            Q-value of the action taken at time t
        :param use_q_target: whether to use the Q-value or the target Q-value
        """
        obs, next_obs = self.obs_to_tensor(obs)[0], self.obs_to_tensor(next_obs)[0]
        with th.no_grad():
            actions = th.as_tensor(actions, device=self.device, dtype=th.int64).unsqueeze(dim=1)
            q = th.gather(self.q_predictor(obs), dim=1, index=actions)
            q_min: th.Tensor = q - th.gather(self.delta_qmin_predictor(obs), 1, actions)
            q_max = q + th.gather(self.delta_qmax_predictor(obs), 1, actions)
            lambda_t1 = ratio(q_min, q, q_max).squeeze(dim=1)
            # If q_max == q_min, we arbitrary set lambda to 0.5 as this should not matter
            lambda_t1[(q_max == q_min).squeeze(dim=1)] = 0.5
            next_q = self.q_target_predictor(next_obs) if use_q_target else self.q_predictor(next_obs)
            self.aspiration = interpolate(next_q.min(dim=1).values, lambda_t1, next_q.max(dim=1).values).cpu().numpy()
            if aspiration_diffs is not None:
                self.aspiration += aspiration_diffs

    def reset_aspiration(self, dones: Optional[np.ndarray] = None) -> None:
        """
        Reset the current aspiration to the initial one

        :param dones: if not None, reset only the aspiration that correspond to the done environments
        """
        if dones is None or isinstance(self.aspiration, float):
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

    def q_values(self, obs: np.ndarray) -> th.Tensor:
        return self.q_predictor(self.obs_to_tensor(obs)[0])

    def action_value(self, obs: np.ndarray, action: np.ndarray):
        return self.q_predictor(obs).gather(1, th.tensor(action, device=self.device, dtype=th.long).unsqueeze(1)).squeeze(1)

    def lambda_ratio(self, obs: np.ndarray, aspiration: Union[float, np.ndarray]) -> th.Tensor:
        q = self.q_values(obs)
        q_min = q.min(dim=1).values
        q_max = q.max(dim=1).values
        lambdas = ratio(q_min, th.tensor(aspiration, device=self.device), q_max)
        lambdas[q_max == q_min] = 0.5  # If q_max == q_min, we set lambda to 0.5, this should not matter
        return lambdas
