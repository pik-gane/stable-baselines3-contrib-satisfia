import contextlib
import os
import sys
from multiprocessing import Process


class TensorboardSupervisor:
    def __init__(self, log_dp):
        self.server = TensorboardServer(log_dp)
        self.server.start()
        print("Started Tensorboard Server")
        self.browser = BrowserProcess()
        print("Started Browser")
        self.browser.start()

    def finalize(self):
        if self.server.is_alive():
            print('Killing Tensorboard Server')
            self.server.terminate()
            self.server.join()
        # As a preference, we leave chrome open - but this may be amended similar to the method above


class TensorboardServer(Process):
    def __init__(self, log_dp):
        super().__init__()
        self.os_name = os.name
        self.log_dp = str(log_dp)
        # self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'{sys.executable} -m tensorboard.main --logdir "{self.log_dp}" 2> NUL')
        elif self.os_name == 'posix':  # Linux
            os.system(f'tensorboard --logdir "{self.log_dp}" 2> /dev/null')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')


class BrowserProcess(Process):
    def __init__(self):
        super().__init__()
        self.os_name = os.name
        self.daemon = True

    def run(self):
        if self.os_name == 'nt':  # Windows
            os.system(f'start chrome  http://localhost:6006/')
        elif self.os_name == 'posix':  # Linux
            os.system(f'firefox http://localhost:6006/')
        else:
            raise NotImplementedError(f'No support for OS : {self.os_name}')


@contextlib.contextmanager
def tensorboard_window(log_dp):
    tb = TensorboardSupervisor(log_dp)
    try:
        yield tb
    finally:
        tb.finalize()


def open_tensorboard(log_dp):
    return TensorboardSupervisor(log_dp)
