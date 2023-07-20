import contextlib
import os
import sys
from multiprocessing import Process
import random
from stable_baselines3.common.callbacks import BaseCallback
import torch as th


class TensorboardSupervisor:
    def __init__(self, log_dp, port=None):
        self.port = port
        if port is None:
            # Choose a random port
            self.port = random.randint(6007, 65535)
        print(f"Using port {self.port}")
        self.server = TensorboardServer(log_dp, self.port)
        self.server.start()
        print("Started Tensorboard Server")
        self.browser = BrowserProcess(self.port)
        print("Started Browser")
        self.browser.start()

    def finalize(self):
        if self.server.is_alive():
            print("Killing Tensorboard Server")
            self.server.terminate()
            self.server.join()
        # As a preference, we leave chrome open - but this may be amended similar to the method above


class TensorboardServer(Process):
    def __init__(self, log_dp, port, ignore_errors=False):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dp)
        self.port = port
        self.ignore_errors = ignore_errors
        # self.daemon = True

    def run(self):
        if self.os_name == "nt":  # Windows
            os.system(
                f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" --port {self.port} {"2> null" if self.ignore_errors else ""}'
            )
        elif self.os_name == "posix":  # Linux
            os.system(f'tensorboard --logdir "{self.log_dp}" --port {self.port} {"2> null" if self.ignore_errors else ""}')
        else:
            raise NotImplementedError(f"No support for OS : {self.os_name}")


class BrowserProcess(Process):
    def __init__(self, port):
        super().__init__()
        self.os_name = os.name
        self.daemon = True
        self.port = port

    def run(self):
        if self.os_name == "nt":  # Windows
            os.system(f"start chrome  http://localhost:{self.port}/")
        elif self.os_name == "posix":  # Linux
            os.system(f"firefox http://localhost:{self.port}/")
        else:
            raise NotImplementedError(f"No support for OS : {self.os_name}")


@contextlib.contextmanager
def tensorboard_window(log_dp):
    tb = TensorboardSupervisor(log_dp)
    try:
        yield tb
    finally:
        tb.finalize()


def open_tensorboard(log_dp):
    return TensorboardSupervisor(log_dp)


class DQNCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0):
        super(DQNCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.logger is not None:
            with th.no_grad():
                q = self.model.q_net(self.model.policy.obs_to_tensor(self.locals["new_obs"])[0])
            self.logger.record_mean("policy/Q_max_mean", float(q.max()))
            self.logger.record_mean("policy/Q_min_mean", float(q.min()))
            self.logger.record_mean("policy/Q_median_mean", float(q.quantile(q=0.5)))
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
