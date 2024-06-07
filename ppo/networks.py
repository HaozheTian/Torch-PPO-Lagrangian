import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

class GaussianPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    
    def forward(self, obs, act=None):
        mean = self.mlp(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        if act == None:
            act = dist.sample()
            log_prob_act = dist.log_prob(act).sum(axis=-1)
            return act, log_prob_act, mean
        else:
            return dist.log_prob(act)


class Value(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    def forward(self, obs):
        return self.mlp(obs)