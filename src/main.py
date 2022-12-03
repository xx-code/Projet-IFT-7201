import gym
import torch
import numpy
from baseline_rl.proximal_policy_optimization import *
from custom_rl.es import ExplorationStrategy
from utils import *
import growspace

class RLgrowspace:

    def __init__(self, gym_env, seed=None):
        self.environment = gym.make(gym_env)
        self.seed = seed

        # apply seed on environment
        self.environment.action_space.seed(seed)
        self.environment.observation_space.seed(seed)
        numpy.random.seed(seed)
        if seed != None:
            torch.manual_seed(seed)
    
    def start_learning_process(self, rl_algo, **kwarg):
        rl_algo.learn(self.environment, **kwarg)

    def visualisation(self, rl_algo):
        obs = self.environment.reset()
        for _ in range(1000):
            action, _states = rl_algo.choose_action(obs)
            obs, reward, done, info = self.environment.step(action)
            self.environment.render()
            if done:
                obs = self.environment.reset()

        self.environment.close()


if __name__ == '__main__':
    '''
        learning_rate=0.00025,
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=32,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_steps=2500,
        clip_range=(0.05, 0.5),
    '''
    print('start')
    es_algo = ExplorationStrategy()
    ppo_algo = ProximalPolicyOptimization()

    steps_ev = 10
    gen = 5000

    growspace_control_evaluation = RLgrowspace('GrowSpaceEnv-Control-v0')
    growspace_control_evaluation.start_learning_process(
        es_algo,
        n_agent=2, 
        n_generation=gen, 
        learning_rate=0.001, 
        sigma=0.1,
        reward_idt=steps_ev
    )
    means, std, timming, timesteps = es_algo.performance()
    means = np.array(means)
    std = np.array(std)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Resultat performance temporaire')

    x = np.linspace(0, gen, len(means))
    ax1.plot(x, means)
    ax1.fill_between(x, means-std, means+std,  alpha=0.4)
    ax1.set(xlabel='genarations', ylabel=f'rewards by means {steps_ev}')

    x_time = np.linspace(0, timesteps, len(means))
    ax2.plot(x_time, means)
    ax2.fill_between(x_time, means-std, means+std, alpha=0.4)
    ax2.set(xlabel='timesteps', ylabel=f'rewards by means {steps_ev}')

    plt.show()

    
    '''growspace_control_evaluation.start_learning_process(
        ppo_algo,
        times=1000
    )'''


    growspace_control_evaluation.visualisation(es_algo)
