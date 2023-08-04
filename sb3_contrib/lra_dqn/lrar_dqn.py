from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th

from sb3_contrib import LRADQN
from sb3_contrib.common.satisficing.utils import interpolate, optional_actions, ratio


class LRARDQN:
    """
    A class that implements the LRAR-DQN algorithm.

    :param initial_aspiration: The initial aspiration value. (can be changed later)
    :param rho: The aspiration rescaling factor.
    :param gamma: The discount factor.
    :param min_lra_dqn: The minimum LRA-DQN.
    :param max_lra_dqn: The maximum LRA-DQN.

    """

    def __init__(self, initial_aspiration, rho: float, min_lra_dqn: LRADQN, max_lra_dqn: LRADQN, gamma: float = 0.99):
        assert min_lra_dqn.device == max_lra_dqn.device, "Devices must be the same for min and max LRADQN"
        self.device = min_lra_dqn.device
        self.gamma = gamma
        self.rho = rho
        self.min_lra_dqn = min_lra_dqn
        self.max_lra_dqn = max_lra_dqn
        self.lra_min = min_lra_dqn.local_relative_aspiration
        self.lra_max = max_lra_dqn.local_relative_aspiration
        self.aspiration = np.array(initial_aspiration)
        self.initial_aspiration = initial_aspiration

    def q(self, obs: np.ndarray, aspiration, actions: Optional[th.Tensor] = None):
        return th.tensor(aspiration, device=self.device).clamp(
            self.q_min(obs, actions=actions), self.q_max(obs, actions=actions)
        )

    def propagate_aspiration(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration.
        """
        with th.no_grad():
            actions = th.tensor(actions, device=self.device, dtype=th.long).unsqueeze(1)
            q = self.q(obs, self.aspiration, actions)
            v_min = interpolate(self.q_min(next_obs).min(dim=1).values, self.lra_min, self.q_min(next_obs).max(dim=1).values)
            v_max = interpolate(self.q_max(next_obs).min(dim=1).values, self.lra_max, self.q_max(next_obs).max(dim=1).values)
            next_lra = ratio(self.q_min(obs, actions=actions), q, self.q_max(obs, actions=actions))
            delta_rescale = -q.cpu().numpy() / self.gamma + interpolate(v_min, next_lra, v_max).cpu().numpy()
            delta_hard = -rewards / self.gamma
        self.aspiration = self.aspiration / self.gamma + interpolate(delta_hard, self.rho, delta_rescale)

    def reset_aspiration(self, dones: Optional[np.ndarray] = None, initial_aspiration: Optional[float] = None):
        """
        Reset the aspiration to its initial value.

        :param dones: If provided, reset only the aspiration of the dones.
        :param initial_aspiration: If provided, reset the aspiration to this value, which becomes the new initial value.
        """
        if initial_aspiration is not None:
            self.initial_aspiration = initial_aspiration
        if dones is None:
            self.aspiration = np.array(self.initial_aspiration)
        else:
            self.aspiration[dones] = self.initial_aspiration

    def predict(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        with th.no_grad():
            q_values_batch = self.q(obs, self.aspiration)
            aspirations = th.tensor(self.aspiration, device=self.device).squeeze()
            # todo?: using a for loop may be crappy, if it's too slow, we could rewrite this using pytorch
            batch_size = len(list(obs.values())[0]) if isinstance(obs, dict) else len(obs)
            actions = th.zeros(batch_size, dtype=th.int, device=self.device)
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
            return actions.cpu().numpy(), None

    @optional_actions
    def q_min(self, obs: np.ndarray) -> th.Tensor:
        return self.min_lra_dqn.q_net(self.max_lra_dqn.policy.obs_to_tensor(obs)[0])

    @optional_actions
    def q_max(self, obs: np.ndarray) -> th.Tensor:
        return self.max_lra_dqn.q_net(self.max_lra_dqn.policy.obs_to_tensor(obs)[0])
