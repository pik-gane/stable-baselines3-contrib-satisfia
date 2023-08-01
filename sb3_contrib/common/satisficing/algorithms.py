import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium.vector.utils import spaces
from stable_baselines3.common.logger import Logger

from sb3_contrib.common.satisficing.policies import ARQPolicy
from sb3_contrib.common.satisficing.utils import interpolate


class ARQAlgorithm(ABC):
    """
    Abstract class for aspiration rescaling algorithms.
    """

    policy: ARQPolicy
    verbose: int
    device: Union[th.device, str]
    action_space: spaces.Discrete
    logger: Logger
    exploration_rate: float

    def __init__(
        self,
        initial_aspiration: float,
        *args,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        gamma: float,
        rho: float,
        **kwargs,
    ) -> None:
        # if type(policy) != str that means that the policy is being loaded from a saved model
        if policy_kwargs is None:
            policy_kwargs = {}
        for kwarg in ["initial_aspiration", "gamma"]:
            if kwarg in policy_kwargs:
                warnings.warn(
                    f"{kwarg} was passed to the policy kwargs, but it will be overwritten by the algorithm kwargs",
                    UserWarning,
                )
        policy_kwargs["initial_aspiration"] = initial_aspiration
        policy_kwargs["gamma"] = gamma
        super().__init__(*args, policy_kwargs=policy_kwargs, **kwargs)  # pytype: disable=wrong-arg-count

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if self.verbose >= 2:
            print(f"My Q values were: {list(self.policy.q_values(observation).cpu().numpy())}")
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                actions = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                actions = np.array(self.action_space.sample())
            if self.verbose >= 2:
                print(f"I am exploring state {observation} and chose action {int(actions)}")
        else:
            actions, _ = self.policy.predict(observation, state, episode_start, deterministic)
            if self.verbose >= 2:
                print(
                    f"My aspiration is {float(self.policy.aspiration):.2f} and my current state {observation}\n"
                    f"So I chose action {int(actions)}"
                )
        return actions, None

    def propagate_aspiration(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        use_q_target: bool = True,
    ) -> None:
        """
        Rescale the aspiration so that, **in expectation**, the agent will
        get the target aspiration.

        :param obs: observations at time t
        :param actions: actions at time t
        :param rewards: rewards at time t
        :param next_obs: observations at time t+1
        :param use_q_target: whether to use the Q-value or the target Q-value
        """
        self.policy.propagate_aspiration(obs, actions, rewards, next_obs, use_q_target=use_q_target)

    def reset_aspiration(self, dones: Optional[np.ndarray] = None) -> None:
        """
        Reset the current aspiration to the initial one

        :param dones: if not None, reset only the aspiration that correspond to the done environments
        """
        self.policy.reset_aspiration(dones)

    @abstractmethod
    def _update_predictors(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        q_target: th.Tensor,
        delta_qmin_target: th.Tensor,
        delta_qmax_target: th.Tensor,
    ):
        """
        Update the predictors of the policy.

        :param obs: the observed states
        :param actions: the actions taken
        :param q_target: the target Q values
        :param delta_qmin_target: the target delta Qmin values
        :param delta_qmax_target: the target delta Qmax values
        """

    def _get_targets(self, new_obs: th.Tensor, rewards: th.Tensor, dones: th.Tensor, smooth_lambda: th.Tensor):
        """
        Compute the targets for the Q values.

        :param new_obs: the new observations
        :param rewards: the rewards obtained
        :param dones: whether the episode is done or not
        :param smooth_lambda: the smoothed relative local aspiration
        """
        with th.no_grad():
            next_q_values = self.policy.q_target_predictor(new_obs)
            v_min = next_q_values.min(dim=1).values.unsqueeze(1)
            v_max = next_q_values.max(dim=1).values.unsqueeze(1)
            v = interpolate(v_min, smooth_lambda, v_max)
            q_target = rewards + self.policy.gamma * dones.logical_not() * v
            delta_qmin_target = self.policy.gamma * (v - v_min) * dones.logical_not()
            delta_qmax_target = self.policy.gamma * (v_max - v) * dones.logical_not()
            return q_target, delta_qmin_target, delta_qmax_target

    def _learning_step(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        new_obs: th.Tensor,
        dones: th.Tensor,
        smooth_lambdas: th.Tensor,
    ) -> None:
        """
        Perform a learning step.

        :param obs: the observed states
        :param actions: the actions taken
        :param rewards: the rewards obtained
        :param new_obs: the new states
        :param dones: whether an episode is done or not
        :param smooth_lambdas: the smoothed relative local aspirations
        """
        q_target, delta_qmin_target, delta_qmax_target = self._get_targets(new_obs, rewards, dones, smooth_lambdas)
        q_pred = self.policy.q_predictor(obs).gather(1, actions).squeeze()
        self.logger.record_mean("train/mean_q_loss", float(F.mse_loss(q_pred, q_target.squeeze()).mean()))
        delta_qmin = self.policy.delta_qmin_predictor(obs).gather(1, actions).squeeze()
        self.logger.record_mean(
            "train/mean_delta_qmin_loss",
            float(F.mse_loss(delta_qmin.squeeze(), delta_qmin_target.squeeze()).mean()),
        )
        delta_qmax = self.policy.delta_qmax_predictor(obs).gather(1, actions).squeeze()
        self.logger.record_mean(
            "train/mean_delta_qmax_loss",
            float(F.mse_loss(delta_qmax, delta_qmax_target.squeeze()).mean()),
        )
        self._update_predictors(obs, actions, q_target, delta_qmin_target, delta_qmax_target)

    def switch_to_eval(self) -> None:
        """
        Prepare the model to be evaluated
        """
        self.exploration_rate = 0.0
        self.reset_aspiration()

    def _log_policy(self, obs: np.ndarray, new_lambda, aspiration_diff) -> None:
        obs = self.policy.obs_to_tensor(obs)[0]
        q = self.policy.q_predictor(obs)
        self.logger.record_mean("policy/Q_max_mean", float(q.max()))
        self.logger.record_mean("policy/Q_min_mean", float(q.min()))
        self.logger.record_mean("policy/Q_median_mean", float(q.quantile(q=0.5)))
        dq_max = self.policy.delta_qmax_predictor(obs)
        self.logger.record_mean("policy/Q_opt_mean", float((q + dq_max).max()))
        dq_min = self.policy.delta_qmin_predictor(obs)
        self.logger.record_mean("policy/Q_unopt_mean", float((q - dq_min).min()))
        self.logger.record_mean("rollout/lambda_mean", float(new_lambda.mean()))
        self.logger.record_mean("rollout/aspiration_mean", float(self.policy.aspiration.mean()))
        self.logger.record_mean("policy/aspiration_diff_mean", float(aspiration_diff.mean()))

    def _excluded_save_params(self) -> List[str]:
        return [
            *super()._excluded_save_params(),
            "policy.q_predictor",
            "policy.q_target_predictor",
            "policy.delta_qmin_predictor",
            "policy.delta_qmax_predictor",
            "policy.delta_qmin_target_predictor",
            "policy.delta_qmax_target_predictor",
        ]
