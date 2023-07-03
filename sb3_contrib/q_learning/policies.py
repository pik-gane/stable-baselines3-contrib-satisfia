import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule


class TabularPolicy(BasePolicy):
    """
    Policy object that implements a Q-Table.

    :param observation_space: Observation space
    :param action_space: Action space
    """

    q_table: th.Tensor

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
        self.q_table = th.nn.Parameter(
            th.zeros((self.observation_space.n, self.action_space.n), device=self.device, requires_grad=False)
        )

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_table[obs].argmax(dim=1).reshape(-1)
