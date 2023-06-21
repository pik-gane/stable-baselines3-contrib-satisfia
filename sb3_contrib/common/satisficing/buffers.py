import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psutil
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib.common.satisficing.type_aliases import SatisficingReplayBufferSamples


class SatisficingReplayBuffer(ReplayBuffer):
    """
    Same as ReplayBuffer but also stores lambda in the transitions

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
        )
        self.lambdas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_lambdas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
            total_memory_usage = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
                + self.lambdas.nbytes
                + self.next_lambdas.nbytes
            )

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add_with_lambda(
        self,
        obs: np.ndarray,
        lambda_: np.ndarray,
        next_obs: np.ndarray,
        next_lambda: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Same as ReplayBuffer.add, but adapted so that it also stores lambda in the transition
        """
        super().add(obs, next_obs, action, reward, done, infos)
        self.lambdas[self.pos] = np.array(lambda_).copy()
        self.next_lambdas[self.pos] = np.array(next_lambda).copy()

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> SatisficingReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.lambdas[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            self.next_lambdas[batch_inds, env_indices, :],
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return SatisficingReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ):
        """
        This should not be used. Use lambda instead
        """
        raise NotImplementedError("Please use add_with_lambda instead")
