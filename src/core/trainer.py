"""Class for training session."""

class Trainer:

    def __init__(
        self, 
        env_id,
        args,
        robot,
        debug,
        log_dir,
        agent,
    ):
        self._args = args
        self._debug = debug
        self._log_dir = log_dir
        self._robot = robot
        self._env_id = env_id
        self._args['robot_model'] = self._robot
        if self._debug:
            self._args['render'] = True
        self._args['debug'] = self._debug
        self._agent = agent(self._env_id, self._args, self._log_dir, self._debug)

    def train(self):
        self._agent.train()

        