from utils import *
import numpy as np
import torch.nn as nn
from torchvision import transforms
import multiprocessing as mp
import torch
import torch.nn.functional as F
from copy import deepcopy
import cv2
import time

DEVICE =  torch.device('cpu')
# NetWork of an agent
class NNModel(torch.nn.Module):
    def __init__(self, out_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        nn.init.normal_(self.conv1.weight, mean=0, std=1.0)
        self.bn1 = nn.BatchNorm2d(16)
        nn.init.normal_(self.bn1.weight)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        nn.init.normal_(self.conv2.weight, mean=0, std=1.0)
        self.bn2 = nn.BatchNorm2d(32)
        nn.init.normal_(self.bn2.weight, mean=0, std=1.0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        nn.init.normal_(self.conv3.weight, mean=0, std=1.0)
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.normal_(self.bn3.weight, mean=0, std=1.0)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(82)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(82)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, out_dim)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(DEVICE)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
class Agent:
    
    def __init__(self, model):
        self.network = model
        self.network.to(DEVICE)
        self.network_size = self.get_weight_size()

    def __transform_input(self, x):
        transform = transforms.Compose(
                        [   transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        return transform(x)[None, :, :, :]
    def get_action(self, x):
        return torch.argmax(self.network.forward(self.__transform_input(x)), dim=1) 
    
    def get_noise_action(self, noise_weight, x):
        copy_agent = deepcopy(self)
        copy_agent.set_weights(noise_weight)

        action = copy_agent.get_action(x)

        del copy_agent

        return action

    def get_weight_size(self):
        shape = 0
        for param in self.network.parameters():
            shape += param.flatten().size()[0]
        return int(shape)

    def get_weights(self):
        arr = []
        for param in self.network.parameters():
            arr.append(param.flatten().cpu().clone()) 
        return torch.concat(arr)

    def set_weights(self, weights):
        if not weights.is_cuda: weights.to(DEVICE)
        start = 0
        for param in self.network.parameters():
            size = param.flatten().size()[0]
            param.data = torch.nn.parameter.Parameter(weights[start: start+size].reshape(param.size())).to(DEVICE)
            start = size
    
    def update_weight(self):
        pass


def get_reward(env, sigma, agent, choose_action, size, seed=None, test=False):
    if seed is not None:
        torch.random.manual_seed(seed)
    # add noise in weight when we learn
    if agent is not None:
        w = agent.get_weights()
        n = torch.randn(size)
        noise = w + sigma * n

    obs = env.reset()
    ep_reward = 0
    while True:
        # choose action
        a = not test if choose_action(obs) else agent.get_noise_action(noise, obs)
        obs, reward, done, _ = env.step(a)
        ep_reward += reward
        if done: break
    return ep_reward

class ExplorationStrategy(RLAlgo):

    def __init__(self):
        self.agent = None
    
    def __train_network(self, env, pool, seed=None):
        # noise_seed = np.random.randint(0, 2**32 - 1, size=self.agent, dtype=np.uint32).repeat(2)

        # training sample in parallel
        # 42 for test
        model = deepcopy(self.agent.network)
        jobs = [
                pool.apply_async(
                    get_reward, 
                    args=(env, self.sigma, Agent(model), self.choose_action, self.agent.network_size, seed))
                    for _ in range(self.n_agent)
            ]
        
        rewards = np.array([j.get() for j in jobs])

        A = (rewards - np.mean(rewards)) / np.std(rewards)
        A = torch.tensor(A)[:, None].float()
        A.to(DEVICE)
        # error
        reconstruction_noise = torch.randn(self.n_agent, self.agent.network_size)
        reconstruction_noise = reconstruction_noise.T
        reconstruction_noise.to(DEVICE)
        cumulative_update = reconstruction_noise.mm(A)
        cumulative_update = cumulative_update.flatten()
        
        current_weight = self.agent.get_weights()

        new_weight = current_weight + self.lr/(self.n_agent*self.sigma) * cumulative_update

        return new_weight, rewards 


    def learn(self, environment, **kwargs):
        self.n_agent = kwargs['n_agent']
        self.n_generation = kwargs['n_generation']
        self.lr = kwargs['learning_rate']
        self.sigma = kwargs['sigma']
        self.steps_record = kwargs['reward_idt']

        assert self.steps_record >= 2, 'must be greater or equale to 2'

        n_core = mp.cpu_count()-1

        pool = mp.Pool(processes=n_core)

        model = NNModel(environment.action_space.n)
        self.agent = Agent(model)

        self.g = []
        self.timming = []
        self.times_of_train = time.time()
        for genaration in range(self.n_generation):
            start_time = time.time()
            new_weight, _ = self.__train_network(environment, pool, seed=42)

            torch.cuda.empty_cache()

            # update_weight
            self.agent.set_weights(new_weight)

            # test mode
            self.g.append(get_reward(environment, self.sigma, self.agent, self.choose_action, self.agent.network_size, seed=None, test=True)) 
            end_time = time.time()
            timestep = end_time - start_time
            self.timming.append(timestep)
            if genaration%self.steps_record == 0:
                print(f'g{genaration}: mean rewards => {np.mean(self.g)}')
                print(f'time { self.timming[-1]}')
        self.times_of_train = time.time() - self.times_of_train
        self.g = np.array(self.g)
        self.timming = np.array(self.timming)


    def performance(self,):
        mean_by_steps = []
        std_by_steps = []
        start = 0
        while start <= len(self.g):
            mean_by_steps.append(np.mean(self.g[start:start+self.steps_record]))
            std_by_steps.append(np.std(self.g[start:start+self.steps_record]))
            start += self.steps_record
        return mean_by_steps, std_by_steps, self.timming, self.times_of_train


    
    def choose_action(self, observation):
        self.agent.get_action(observation)