from gym.envs.registration import register

register(
    id='gym_flappybird/FlappyBird-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)