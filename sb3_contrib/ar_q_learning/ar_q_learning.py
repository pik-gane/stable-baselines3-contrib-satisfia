import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium.vector.utils import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, safe_mean
from torch.nn import functional as F
from torch.nn.functional import relu

from sb3_contrib.ar_q_learning.policies import ARQLearningPolicy, QTable, DeltaQTable
from sb3_contrib.common.satisficing.utils import interpolate, ratio

SelfArQLearning = TypeVar("SelfArQLearning", bound="ArQLearning")


class ARQLearning(BaseAlgorithm):
    """
    Tabular Q-Learning with Aspiration Rescaling
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "TabularPolicy": ARQLearningPolicy,
    }
    exploration_schedule: Schedule
    q_table: QTable
    delta_qmin_table: DeltaQTable
    delta_qmax_table: DeltaQTable
    policy: ARQLearningPolicy

    def __init__(
        self,
        env: Union[GymEnv, str, None],
        policy: Union[str, Type[BasePolicy]] = "TabularPolicy",
        learning_rate: Union[float, Schedule] = 0.5,  # Jobst: maybe learn slower by default?
        mu: float = 0.0,
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
        # if type(policy) != str that means that the policy is being loaded from a saved model
        if not ((type(policy) != str) or (policy_kwargs is not None and "initial_aspiration" in policy_kwargs.keys())):
            warnings.warn(
                "Aspiration rescaling Q learning requires a value for initial_aspiration in policy_kwargs"
                "\nIf this is not a test, consider setting this parameter",
                UserWarning,
            )
            if policy_kwargs is None:
                policy_kwargs = {}
            policy_kwargs["initial_aspiration"] = 10.0  # Jobst: why 10.0? Maybe 0.0 is more natural?
        super().__init__(
            ARQLearningPolicy,
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
        self.mu = mu
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
        self.delta_qmin_table = self.policy.delta_qmin_table
        self.delta_qmax_table = self.policy.delta_qmax_table

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
            aspiration_diff = self.policy.aspiration - self.policy.q_table(observation, action).cpu().numpy()
            if self.verbose >= 2:
                print(f"I am exploring state {int(observation)} and chose action {int(action)}")
        else:
            action, aspiration_diff = self.policy.predict(observation, state, episode_start, deterministic)
            if self.verbose >= 2:
                print(
                    f"My aspiration is {float(self.policy.aspiration)} and my current state {int(observation)}\n"
                    f"My Q values were: {list(self.policy.q_table(observation)[0].cpu().numpy())} so I chose action {int(action)}"
                )
        return action, aspiration_diff

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfArQLearning:
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
        last_lambda = self.policy.lambda_ratio(obs, aspiration)
        while self.num_timesteps < total_timesteps:
            num_collected_steps += 1
            if self.verbose >= 2:
                debug_len = len(f"==============Step {num_collected_steps}============\n")
                print(f"==============Step {num_collected_steps}============\n")
            self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
            learning_rate = self.lr_schedule(self._current_progress_remaining)
            # Step
            actions, aspiration_diff = self.predict(obs)
            new_obs, rewards, dones, infos = self.env.step(actions)
            if self.verbose >= 2:
                print(f"I receive reward {float(rewards)} and arrive in state {int(new_obs)}\n")
            self.rescale_aspiration(obs, actions, new_obs, aspiration_diff=aspiration_diff)
            self.reset_aspiration(dones)
            new_lambda = self.policy.lambda_ratio(new_obs, self.policy.aspiration)

            if self.verbose >= 2:
                actions = th.as_tensor(actions, device=self.device, dtype=th.int64).unsqueeze(dim=1)
                q = th.gather(self.q_table(obs), dim=1, index=actions)
                q_min = q - th.gather(self.delta_qmin_table(obs), 1, actions)
                q_max = q + th.gather(self.delta_qmax_table(obs), 1, actions)
                lambda_t1 = float(ratio(q_min, q, q_max))
                print(
                    f"\nMy delta Qmin was {list(self.policy.delta_qmin_table(obs)[0].cpu().numpy())} and my delta Qmax was {list(self.policy.delta_qmax_table(obs)[0].cpu().numpy())}\n"
                    f"My new lambda is therefore {lambda_t1} and my new aspiration is {float(self.policy.aspiration)}"
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
            # Update Q tables
            smooth_lambda = interpolate(new_lambda, self.mu, last_lambda)
            smooth_lambda[dones] = new_lambda  # For the ended episodes, we must ignore the last lambda
            with th.no_grad():
                v_min = self.q_table(new_obs).min(dim=1).values
                v_max = self.q_table(new_obs).max(dim=1).values
                v = interpolate(v_min, smooth_lambda, v_max)
                r = th.tensor(rewards, device=self.device)
                t_dones = th.tensor(dones, device=self.device)
                q_target = r + self.gamma * t_dones.logical_not() * v
                delta_qmin_target = relu(self.gamma * (v - v_min)) * t_dones.logical_not()
                delta_qmax_target = relu(self.gamma * (v_max - v)) * t_dones.logical_not()
                if self.verbose >= 2:
                    print()
                    if not dones.any():
                        print(
                            f"In state {int(new_obs)}, my v_min is {float(v_min)} and my v_max is {float(v_max)}\n"
                            f"My new_lambda is {float(new_lambda)} and my smooth_lambda is {float(smooth_lambda)}\n"
                            f"Therefore my v is {float(v)} and my q_target is {float(q_target)}\n"
                            f"My delta_qmin_target is {float(delta_qmin_target)} and my delta_qmax_target is {float(delta_qmax_target)}\n"
                        )
                    print(
                        f"\nNow I perform my update step:\n - q[{int(obs)}, {int(actions)}] = {float(q)} + "
                        f"{float(learning_rate)} * ({float(q_target)} - {float(q)}) = {float(q + learning_rate * (q_target - self.q_table(obs, actions)))}\n"
                        f" - delta_qmin[{int(obs)}, {int(actions)}] = {float(self.policy.delta_qmin_table(obs, actions))} + "
                        f"{float(learning_rate)} * ({float(delta_qmin_target)} - {float(self.policy.delta_qmin_table(obs, actions))}) = {float(self.policy.delta_qmin_table(obs, actions) + learning_rate * (delta_qmin_target - self.policy.delta_qmin_table(obs, actions)))}\n"
                        f" - delta_qmax[{int(obs)}, {int(actions)}] = {float(self.policy.delta_qmax_table(obs, actions))} + "
                        f"{float(learning_rate)} * ({float(delta_qmax_target)} - {float(self.policy.delta_qmax_table(obs, actions))}) = {float(self.policy.delta_qmax_table(obs, actions) + learning_rate * (delta_qmax_target - self.policy.delta_qmax_table(obs, actions)))}"
                    )
                self.q_table.update_table(obs, actions, q_target, learning_rate)
                self.delta_qmin_table.update_table(obs, actions, delta_qmin_target, learning_rate)
                self.delta_qmax_table.update_table(obs, actions, delta_qmax_target, learning_rate)
                self.logger.record_mean(
                    "train/mean_q_loss", float(F.mse_loss(self.q_table(obs, actions).squeeze(dim=-1), q_target).mean())
                )
                self.logger.record_mean(
                    "train/mean_delta_qmin_loss",
                    float(F.mse_loss(self.delta_qmin_table(obs, actions).squeeze(dim=-1), delta_qmin_target).mean()),
                )
                self.logger.record_mean(
                    "train/mean_delta_qmax_loss",
                    float(F.mse_loss(self.delta_qmax_table(obs, actions).squeeze(dim=-1), delta_qmax_target).mean()),
                )
                self.logger.record_mean("rollout/lambda_mean", float(new_lambda.mean()))
                self.logger.record_mean("rollout/aspiration_mean", float(self.policy.aspiration.mean()))
                q = self.q_table(obs)
                self.logger.record_mean("policy/Q_max_mean", float(q.max()))
                self.logger.record_mean("policy/Q_min_mean", float(q.min()))
                self.logger.record_mean("policy/Q_median_mean", float(q.quantile(q=0.5)))
                self.logger.record_mean("policy/Q_opt_mean", float((q + self.delta_qmax_table(obs)).max()))
                self.logger.record_mean("policy/Q_unopt_mean", float((q - self.delta_qmin_table(obs)).min()))
                # Print all information about the update. Observation, action, reward, lambda, aspiration, Q, and Q updates
                # in a single print statement in a nice table
                if self.verbose >= 2:
                    print("=" * debug_len + "\n")
            obs = new_obs
            last_lambda = new_lambda
        callback.on_training_end()
        return self

    def _on_step(self):
        self.num_timesteps += self.env.num_envs
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def rescale_aspiration(
        self,
        obs_t: np.ndarray,
        a_t: np.ndarray,
        obs_t1: np.ndarray,
        dones: Optional[np.ndarray] = None,
        aspiration_diff: Optional[np.ndarray] = None,
    ) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration.

        :param obs_t: observations at time t
        :param a_t: actions at time t
        :param obs_t1: observations at time t+1
        """
        self.policy.rescale_aspiration(obs_t, a_t, obs_t1, dones, aspiration_diff)

    def reset_aspiration(self, dones: Optional[np.ndarray] = None) -> None:
        """
        Reset the current aspiration to the initial one

        :param dones: if not None, reset only the aspiration that correspond to the done environments
        """
        self.policy.reset_aspiration(dones)

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

    def switch_to_eval(self) -> None:
        """
        Prepare the model to be evaluated
        """
        self.exploration_rate = 0.0
        self.reset_aspiration()

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_table"]
