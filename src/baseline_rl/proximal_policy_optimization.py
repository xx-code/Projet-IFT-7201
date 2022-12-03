from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from utils import RLAlgo

class ProximalPolicyOptimization(RLAlgo):

    def learn(self, environment, *wargs, **kwargs):
        '''
            # timesteps : int number of epoch for learning
            learning_rate=kwargs['learning_rate'],
            gamma=kwargs['gamma'],
            gae_lambda=kwargs['gae_lambda'],
            batch_size=kwargs['batch_size'],
            ent_coef=kwargs['ent_coef'],
            max_grad_norm=kwargs['max_grad_norm'],
            n_steps=kwargs['n_steps'],
            clip_range=kwargs['clip_range'],
        '''
        self.model = PPO(
            'MlpPolicy', 
            environment,
            verbose=1)
        self.model.learn(total_timesteps=kwargs['times'])

    def evaluation(self):
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=10)
        return mean_reward, std_reward

    def choose_action(self, observation):
        return self.model.predict(observation)