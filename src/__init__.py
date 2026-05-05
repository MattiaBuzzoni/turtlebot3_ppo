from gym.envs.registration import register

register(
    id="GoEnv",
    entry_point="src.gym.env.got_to.go_env:GoEnv",
)