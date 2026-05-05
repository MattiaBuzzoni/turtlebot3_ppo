from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(
        self,
        env_id,
        env_args,
        log_dir,
        debug_mode,
    ):
        self._env_id = env_id
        self._env_args = env_args
        self._debug = debug_mode
        self._log_dir = log_dir

    @abstractmethod
    def train(self):
        pass