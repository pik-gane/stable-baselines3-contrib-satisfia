import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, safe_mean

from sb3_contrib.ar_q_learning.policies import ARQLearningPolicy, DeltaQTable, QTable
from sb3_contrib.common.satisficing.algorithms import ARAlgorithm
from sb3_contrib.common.satisficing.utils import interpolate, ratio

SelfARQLearning = TypeVar("SelfARQLearning", bound="ARQLearning")


class ARQLearning(ARAlgorithm, BaseAlgorithm):
    """
    Tabular Q-Learning with Aspiration Rescaling
    """

    policy_aliases: Dict[str, Type[ARQLearningPolicy]] = {
        "ARQLearningPolicy": ARQLearningPolicy,
    }
    exploration_schedule: Schedule
    q_table: QTable
    delta_qmin_table: DeltaQTable
    delta_qmax_table: DeltaQTable
    policy: ARQLearningPolicy

    def __init__(
        self,
        env: Union[GymEnv, str, None],
        policy: Union[str, Type[ARQLearningPolicy]] = "ARQLearningPolicy",
        learning_rate: Union[float, Schedule] = 0.1,
        mu: float = 0.5,
        gamma: float = 0.99,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "cpu",
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            gamma=gamma,
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
        self.mu = mu
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
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
        self.delta_qmin_table = self.policy.delta_qmin_table
        self.delta_qmax_table = self.policy.delta_qmax_table

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfARQLearning:
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
        aspiration = self.policy.initial_aspiration
        last_lambda = self.policy.lambda_ratio(obs, aspiration).clamp(min=0, max=1)
        while self.num_timesteps < total_timesteps:
            num_collected_steps += 1
            if self.verbose >= 2:
                debug_len = len(f"==============Step {num_collected_steps}============\n")
                print(f"==============Step {num_collected_steps}============\n")
            self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
            self.learning_rate = self.lr_schedule(self._current_progress_remaining)
            # Step
            actions, _ = self.predict(obs)
            new_obs, rewards, dones, infos = self.env.step(actions)
            if self.verbose >= 2:
                print(f"I receive reward {float(rewards):.2f} and arrive in state {int(new_obs)}\n")
            with th.no_grad():
                aspiration_diff = self.policy.aspiration - np.take_along_axis(
                    self.policy.q_values(obs).cpu().numpy(), np.expand_dims(actions, 1), 1
                ).squeeze(1)
            self.rescale_aspiration(obs, actions, new_obs)
            self.reset_aspiration(dones)
            new_lambda = self.policy.lambda_ratio(new_obs, self.policy.aspiration).clamp(min=0, max=1)
            if self.verbose >= 2:
                t_actions = th.as_tensor(actions, device=self.device, dtype=th.int64).unsqueeze(dim=1)
                q = th.gather(self.q_table(obs), dim=1, index=t_actions)
                q_min = q - th.gather(self.delta_qmin_table(obs), 1, t_actions)
                q_max = q + th.gather(self.delta_qmax_table(obs), 1, t_actions)
                lambda_t1 = float(ratio(q_min, q, q_max))
                print(
                    f"\nMy delta Qmin was {float(q_min):.2f} and my delta Qmax was {float(q_max):.2f}.\n"
                    f"My new lambda is therefore {float(q_min):.2f}:{float(q):.2f}:{float(q_max):.2f} = {lambda_t1:.2g}.\n"
                    f"As my Q values in next state {int(new_obs)} are {list(self.q_table(new_obs)[0].cpu().numpy())}\n"
                    f"my new aspiration is {float(self.policy.aspiration - aspiration_diff):.2g} + {float(aspiration_diff):.2f} = {float(self.policy.aspiration):.2f}\n"
                    + f"And my new (clamped) lambda is therefore: {float(new_lambda):.2f}"
                )
            # Collect logs
            self._on_step()
            self._update_info_buffer(infos, dones)
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            num_collected_episodes += dones.sum()
            self._episode_num += dones.sum()
            if log_interval is not None and self._episode_num - last_log_time >= log_interval:
                self._dump_logs()
                last_log_time = self._episode_num
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                break
            smooth_lambda = interpolate(new_lambda, self.mu, last_lambda)
            smooth_lambda[dones] = new_lambda  # For the ended episodes, we must ignore the last lambda
            # Update Q tables
            actions, dones = th.tensor(actions, device=self.device), th.tensor(dones, device=self.device)
            obst, new_obs = th.as_tensor(obs, device=self.device), th.as_tensor(new_obs, device=self.device)
            rewards = th.as_tensor(rewards, device=self.device)
            smooth_lambda = th.as_tensor(smooth_lambda, device=self.device)
            actions = th.as_tensor(actions, device=self.device, dtype=th.long).unsqueeze(dim=1)
            self._learning_step(obst, actions, rewards, new_obs, dones, smooth_lambda)
            # if self.verbose >= 2: todo remove
            #     print()
            #     if not dones.any():
            #         print(
            #             f"In state {int(new_obs)}, my v_min is {float(v_min):.2f} and my v_max is {float(v_max):.2f}\n"
            #             f"My new_lambda is {float(new_lambda):.2f} and my smooth_lambda is {float(smooth_lambda):.2f}\n"
            #             f"Therefore my v is {float(v):.2f} and my q_target is {float(q_target):.2f}\n"
            #             f"My delta_qmin_target is {float(delta_qmin_target):.2f} and my delta_qmax_target is {float(delta_qmax_target):.2f}\n"
            #         )
            #     print(
            #         f"\nNow I perform my update step:\n - q[{int(obs)}, {int(actions)}] = {float(q):.2f} + "
            #         f"{float(learning_rate):.2f} * ({float(q_target):.2f} - {float(q):.2f}) = {float(q + learning_rate * (q_target - self.q_table(obs, actions))):.2g}\n"
            #         f" - delta_qmin[{int(obs)}, {int(actions)}] = {float(self.policy.delta_qmin_table(obs, actions)):.2f} + "
            #         f"{float(learning_rate):.2f} * ({float(delta_qmin_target):.2f} - {float(self.policy.delta_qmin_table(obs, actions)):.2f}) = {float(self.policy.delta_qmin_table(obs, actions) + learning_rate * (delta_qmin_target - self.policy.delta_qmin_table(obs, actions))):.2g}\n"
            #         f" - delta_qmax[{int(obs)}, {int(actions)}] = {float(self.policy.delta_qmax_table(obs, actions)):.2f} + "
            #         f"{float(learning_rate):.2f} * ({float(delta_qmax_target):.2f} - {float(self.policy.delta_qmax_table(obs, actions)):.2f}) = {float(self.policy.delta_qmax_table(obs, actions) + learning_rate * (delta_qmax_target - self.policy.delta_qmax_table(obs, actions))):.2g}"
            #     )
            self._log_policy(obs, new_lambda, aspiration_diff)
            if self.verbose >= 2:
                print("=" * debug_len + "\n")
            obs = new_obs
            last_lambda = new_lambda
        callback.on_training_end()
        return self

    def _update_predictors(self, obs, actions, q_target, delta_qmin_target, delta_qmax_target):
        learning_rate = self.learning_rate
        self.q_table.update_table(obs, actions, q_target, learning_rate)
        self.delta_qmin_table.update_table(obs, actions, delta_qmin_target, learning_rate)
        self.delta_qmax_table.update_table(obs, actions, delta_qmax_target, learning_rate)

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
        return [*super()._excluded_save_params(), "q_table", "delta_qmin_table", "delta_qmax_table"]

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Set the environment AND reset it. force_reset will be ignored and the environment will always
        be reset to avoid weird aspirations setups

        :param env: The environment for learning a policy
        :param force_reset: Ignored, the function will always make is if this is True
        """
        if not force_reset:
            warnings.warn(
                UserWarning(
                    "force_reset is ignored in AR Q learning. The environment will always be reset to avoid "
                    "weird aspirations setups"
                )
            )
        super().set_env(env, True)
        # Update the aspiration shape according to the env
        self.policy.reset_aspiration()
