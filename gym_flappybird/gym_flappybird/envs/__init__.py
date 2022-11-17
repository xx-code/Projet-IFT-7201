from gym.envs.registration import register

register(
    id='gym_flappybird/Flappybird-v0',
    entry_point='gym_flappybird.envs:Flappybird',
    max_episode_steps=300,
)