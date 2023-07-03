import warnings
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss, SmoothL1Loss

from sb3_contrib.ar_dqn.policies import ArDQNPolicy, CnnPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from sb3_contrib.ar_dqn.utils import interpolate, ratio
from sb3_contrib.common.satisficing.buffers import SatisficingReplayBuffer
from sb3_contrib.common.satisficing.type_aliases import SatisficingReplayBufferSamples

SelfArDQN = TypeVar("SelfArDQN", bound="ArDQN")


class ArDQN(DQN):
    """
    Deep Q-Network (DQN) with aspiration rescaling (AR)

    Paper: ? based on DQN https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the DQN Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param mu: the aspiration smoothing coefficient (between 0 and 1) default 0 for no smoothing for lambda
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param loss:
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. MUST contains
        initial_aspiration: The inital aspiration of the agent i.e the desired reward over a run
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    loss_aliases: Dict[str, nn.Module] = {"MSE": MSELoss, "SmoothL1Loss": SmoothL1Loss}
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: ArDQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[ArDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 1000,
        batch_size: int = 32,
        mu: float = 0.0,
        tau: float = 1.0,
        gamma: float = 0.99,
        loss: Literal["MSE", "SmoothL1Loss"] = "MSE",
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if not ((type(policy) != str) or (policy_kwargs is not None and "initial_aspiration" in policy_kwargs.keys())):
            warnings.warn(
                "ArDQN requires a value for initial_aspiration in policy_kwargs\nIf this is not a test, consider setting this parameter",
                UserWarning,
            )
            if policy_kwargs is None:
                policy_kwargs = {}
            policy_kwargs["initial_aspiration"] = 10.0

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class=SatisficingReplayBuffer,  # We need dict because we want to store lambdas
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.mu = mu
        self.test_env = deepcopy(self.env)
        self.loss = self.loss_aliases[loss]()

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.delta_qmax_net = self.policy.delta_qmax_net
        self.delta_qmin_net = self.policy.delta_qmin_net

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        q_losses = []
        qmax_losses = []
        qmin_losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data: SatisficingReplayBufferSamples = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                smooth_lambda = interpolate(replay_data.next_lambda, self.mu, replay_data.lambda_).unsqueeze(1)
                v_min = next_q_values.min(dim=1).values.unsqueeze(1)
                v_max = next_q_values.max(dim=1).values.unsqueeze(1)
                v = interpolate(v_min, smooth_lambda, v_max)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * v
                target_delta_min = self.gamma * (v - v_min)
                target_delta_max = self.gamma * (v_max - v_min)

            obs = replay_data.observations
            # Get current Q-values estimates
            current_q_values = self.q_net(obs)
            current_delta_min, current_delta_max = self.delta_qmin_net(obs), self.delta_qmax_net(obs)

            index = replay_data.actions.long()
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=index)
            current_delta_min = th.gather(current_delta_min, dim=1, index=index)
            current_delta_max = th.gather(current_delta_max, dim=1, index=index)

            # Compute Huber loss (less sensitive to outliers)
            q_loss = self.loss(current_q_values, target_q_values)
            qmin_loss = self.loss(current_delta_min, target_delta_min)
            qmax_loss = self.loss(current_delta_max, target_delta_max)

            # Logs
            q_losses.append(q_loss.item())
            qmax_losses.append(qmax_loss.item())
            qmin_losses.append(qmin_loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            # Backward all losses
            (q_loss + qmin_loss + qmax_loss).backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps
        if self.test_env is None:
            self.test_env = deepcopy(self.env)
        with th.no_grad():
            reset_obs, _ = self.policy.obs_to_tensor(self.test_env.reset())
            q = self.q_net(reset_obs)
            self.logger.record_mean("policy/Q_max_mean", float(q.max().cpu()))
            self.logger.record_mean("policy/Q_min_mean", float(q.min()))
            self.logger.record_mean("policy/Q_median_mean", float(q.quantile(q=0.5)))
            self.logger.record_mean("policy/Q_opt_mean", float((q + self.delta_qmax_net(reset_obs)).max()))
            self.logger.record_mean("policy/Q_unopt_mean", float((q - self.delta_qmin_net(reset_obs)).min()))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record_mean("train/mean_q_loss", np.mean(q_losses))
        self.logger.record_mean("train/mean_qmax_loss", np.mean(qmax_losses))
        self.logger.record_mean("train/mean_qmin_loss", np.mean(qmin_losses))
        self.logger.record("train/q_loss", np.mean(q_losses))
        self.logger.record("train/qmax_loss", np.mean(qmax_losses))
        self.logger.record("train/qmin_loss", np.mean(qmin_losses))

    def learn(
        self: SelfArDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "ArDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfArDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def rescale_aspiration(self, obs_t: np.ndarray, a_t: np.ndarray, obs_t1: np.ndarray) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration.

        :param obs_t: observation at time t
        :param a_t: action at time t
        :param obs_t1: observation at time t+1
        """
        self.policy.rescale_aspiration(self.policy.obs_to_tensor(obs_t)[0], a_t, self.policy.obs_to_tensor(obs_t1)[0])

    def reset_aspiration(self, dones: Optional[np.ndarray] = None) -> None:
        """
        Reset the current aspiration to the initial one

        :param dones: if not None, reset only the aspiration that correspond to the done environments
        """
        self.policy.reset_aspiration(dones)

    def switch_to_eval(self) -> None:
        """
        Prepare the model to be evaluated
        """
        self.exploration_rate = 0.0
        self.reset_aspiration()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: SatisficingReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Note: We need to override this method in order to store the lambdas
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Rescale aspiration
            with th.no_grad():
                new_obs_th, _ = self.policy.obs_to_tensor(new_obs)
                # will update self.policy.aspiration
                self.policy.rescale_aspiration(self.policy.obs_to_tensor(self._last_obs)[0], buffer_actions, new_obs_th)
                new_qs = self.q_net(new_obs_th).cpu().numpy()
            self.reset_aspiration(dones)
            new_lambda = ratio(new_qs.min(axis=1), self.policy.aspiration, new_qs.max(axis=1))
            self.logger.record_mean("rollout/lambda_mean", new_lambda.mean())
            self.logger.record_mean("rollout/aspiration_mean", self.policy.aspiration.mean())
            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_ar_transition(replay_buffer, buffer_actions, new_obs, new_lambda, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _store_ar_transition(
        self,
        replay_buffer: SatisficingReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        new_lambda: np.ndarray,
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).
        This is the same as self._store_transition, but it stores the lambda value as well.

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param new_lambda: lambda value for the next observation
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])
        # Todo: Do we need to do something for lambda in case of done ?

        replay_buffer.add_with_lambda(
            self._last_original_obs,
            self._last_lambda,
            next_obs,
            new_lambda,
            buffer_action,
            reward_,
            dones,
            infos,
        )
        self._last_lambda = new_lambda
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def _excluded_save_params(self) -> List[str]:
        return [
            *super()._excluded_save_params(),
            "delta_qmin_net",
            "delta_qmax_net",
        ]

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        # check if _last_obs and env will be reset in the super call of base_class
        reset = reset_num_timesteps or self._last_obs is None
        r = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        if reset:
            training_mode = self.policy.training
            self.policy.set_training_mode(False)
            # Initialize the lambda value
            with th.no_grad():
                q = self.q_net(th.tensor(self._last_obs, device=self.device)).cpu().numpy()
            self._last_lambda = ratio(q.min(axis=1), self.policy.initial_aspiration, q.max(axis=1))
            self.reset_aspiration()
            self.policy.set_training_mode(training_mode)
        return r

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Set the environment AND reset it. force_reset will be ignored and the environment will always
        be reset to avoid weird aspirations setups

        :param env: The environment for learning a policy
        :param force_reset: Ignored, the function will always make is if this is True
        """
        super().set_env(env, True)
        # Update the aspiration shape according to the env
        self.policy.reset_aspiration()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        To make a prediction using the AR DQN, set deterministic=False and exploration_rate=0.
        The version with deterministic=True is exposed only for testing purposes
        """
        return super().predict(observation, state, episode_start, deterministic)
