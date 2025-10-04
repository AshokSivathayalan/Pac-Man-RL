from gymnasium.envs.registration import register

register(
    id="pacman_env/GridWorld-v0",
    entry_point="pacman_env.envs:GridWorldEnv",
)
