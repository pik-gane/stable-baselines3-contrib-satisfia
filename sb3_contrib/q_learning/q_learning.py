import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, safe_mean
from torch.nn import functional as F

from sb3_contrib.q_learning.policies import TabularPolicy

SelfQLearning = TypeVar("SelfQLearning", bound="QLearning")


class QLearning(BaseAlgorithm):
    """
    Tabular Q-Learning
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "TabularPolicy": TabularPolicy,
    }
    exploration_schedule: Schedule
    q_table: th.Tensor
    policy: TabularPolicy

    def __init__(
        self,
        # policy: Union[str, Type[BasePolicy]], for now we don't need it
        env: Union[GymEnv, str, None],
        learning_rate: Union[float, Schedule] = 0.5,
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            TabularPolicy,
            env,
            learning_rate,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=False,
            seed=seed,
            use_sde=False,
            supported_action_spaces=(spaces.Discrete,),
        )
        self.gamma = gamma
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0
        if self.env is not None:
            assert isinstance(
                self.observation_space,
                spaces.Discrete,
            ), f"Q learning only support {spaces.Discrete} as observation but {self.observation_space} was provided"
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.q_table = self.policy.q_table

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfQLearning:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())
        obs = self.env.reset()
        num_collected_steps, num_collected_episodes = 0, 0
        last_log_time = 0
        while self.num_timesteps < total_timesteps:
            self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
            learning_rate = self.lr_schedule(self._current_progress_remaining)

            action = self.predict(obs)[0]
            new_obs, rewards, dones, infos = self.env.step(action)
            self._on_step()
            self._update_info_buffer(infos, dones)
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            num_collected_steps += dones.sum()
            self._episode_num += dones.sum()
            if log_interval is not None and self._episode_num - last_log_time >= log_interval:
                self._dump_logs()
                last_log_time = self._episode_num
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                break
            with th.no_grad():
                q_target = (
                    th.tensor(rewards, device=self.device)
                    + self.gamma * th.tensor(dones, device=self.device).logical_not() * self.q_table[new_obs].max(dim=1).values
                )
                q = self.q_table[obs, action]
                self.q_table[obs, action] += learning_rate * (q_target - q)
                self.logger.record_mean("train/mean_q_loss", float(F.mse_loss(q, q_target).mean()))
            obs = new_obs
        callback.on_training_end()
        return self

    def _on_step(self):
        self.num_timesteps += self.env.num_envs
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_table"]
