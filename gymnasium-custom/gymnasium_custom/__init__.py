from gymnasium.envs.registration import register

register(
    id="ICTesting-v0",
    entry_point="gymnasium_custom.envs:IcTestEnvironment",
    max_episode_steps=2749,
)
