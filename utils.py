import contextlib
import os
import sys
from multiprocessing import Process
import random


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
