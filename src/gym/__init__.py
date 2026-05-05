from gymnasium.envs.registration import register

register(
    id="GoEnv",
    entry_point="src.gym.env.go_to.go_env:GoEnv",
)