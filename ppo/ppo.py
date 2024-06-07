import torch
import numpy as np
import gymnasium as gym
from torch.nn.functional import softplus
from torch.optim import Adam
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ppo.networks import GaussianPolicy, Value
from ppo.buffers import PPOBuffer, PPOLagBuffer, PPOBufferItem, PPOLagBufferItem
from ppo.utils import obs2ten, ten2arr, Logger

from typing import Union



class PPO:
    def __init__(self, env, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on {self.device}')
        self.env = env
        self.env_name = env.__class__.__name__
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
        self._init_hyperparameters(**kwargs)
        self._init_seed()
        self._init_networks()
        self.buffer = PPOBuffer(self.device, self.gam)
        self.logger = Logger()
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/PPO_'+self.env_name+self.time_str)

    def learn(self):
        self.num_eps = 0
        step = 0
        with tqdm(total=self.total_steps) as pbar:
            while step < self.total_steps:
                # get episodes
                data, epoch_steps = self.rollout()
                # update networks
                self.update_policy(data)
                self.update_value_func(data)
                # record the number of steps in the epoch
                step += epoch_steps
                pbar.update(epoch_steps)

    def rollout(self) -> Union[PPOBufferItem, int]:
        epoch_step = 0
        while epoch_step < self.min_epoch_steps:
            obs, info = self.env.reset(seed=self.seed)
            eps_ret, eps_cost, eps_len = 0, 0, 0  # episode record
            while True:
                act, logp, _ = self.get_action(obs)
                obs_next, rew, term, trun, info = self.env.step(act)
                self.buffer.add(obs, act, rew, self.evaluate(obs), logp)

                obs = obs_next
                eps_ret, eps_cost, eps_len = eps_ret+rew, eps_cost+info["cost"], eps_len+1

                if term or trun:
                    last_val = 0 if term else self.evaluate(obs)
                    self.buffer.path_done(last_val)

                    epoch_step += eps_len
                    self.num_eps += 1
                    eps_info = {'eps_ret': eps_ret, 'eps_cost': eps_cost, 'eps_len': eps_len}
                    self.logger.add(eps_info)
                    if self.use_tb:
                        self._to_tb(eps_info)
                    break
        return self.buffer.get(), epoch_step
            
    def get_action(self, obs: np.ndarray) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            act, logp, mean = self.policy(obs2ten(obs, self.device))
        return ten2arr(act), ten2arr(logp), ten2arr(mean)
    
    def evaluate(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            v_obs = self.value(obs2ten(obs, self.device))
        return ten2arr(v_obs)[0]
    
    def update_policy(self, data: PPOBufferItem):
        adv = data.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        for _ in range(self.num_updates):
            log_probs = self.policy(data.observations, data.actions).squeeze()
            ratio = torch.exp(log_probs - data.log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_policy = (-torch.min(surr1, surr2)).mean()

            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            self.policy_optimizer.step()

    def update_value_func(self, data: PPOBufferItem):
        for _ in range(self.num_updates):
            loss_val = ((self.value(data.observations).squeeze() - data.reward_to_gos)**2).mean()
            self.value_optimizer.zero_grad()
            loss_val.backward()
            self.value_optimizer.step()

    def _init_hyperparameters(self, **kwargs):
        self.seed = kwargs.get('seed', 0)
        self.min_epoch_steps = kwargs.get('min_epoch_steps', 1000)
        self.total_steps = kwargs.get('total_steps', 300000)
        self.clip_ratio = kwargs.get('clip_ratio', 0.2)
        self.num_updates = kwargs.get('num_updates', 5)
        self.gam = kwargs.get('gamma', 0.95)
        self.policy_lr = kwargs.get('policy_lr', 0.005)
        self.value_lr = kwargs.get('v_lr', 0.005)
        self.use_tb = kwargs.get('use_tb', False)
    
    def _init_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    def _init_networks(self):
        self.policy = GaussianPolicy(self.env).to(self.device)
        self.value = Value(self.env).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=self.value_lr)
    
    def _to_tb(self, eps_info):
        for name, scalar in eps_info.items():
            self.writer.add_scalar(f'charts/{name}', scalar, self.num_eps)



class PPOLag(PPO):
    def __init__(self, env, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Running on {self.device}')
        self.env = env
        self.env_name = env.__class__.__name__
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
        self._init_hyperparameters(**kwargs)
        self._init_seed()
        self._init_networks()
        self.buffer = PPOLagBuffer(self.device, self.gam)
        self.logger = Logger()
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/PPOLag_'+self.env_name+self.time_str)

    def rollout(self) -> Union[PPOLagBufferItem, int]:
        epoch_step = 0
        while epoch_step < self.min_epoch_steps:
            obs, info = self.env.reset(seed=self.seed)
            eps_ret, eps_cost, eps_len = 0, 0, 0  # episode record
            while True:
                act, logp, _ = self.get_action(obs)
                obs_next, rew, term, trun, info = self.env.step(act)

                cost = info["cost"]
                val, val_cost = self.evaluate(obs)
                self.buffer.add(obs, act, rew, val, logp, cost, val_cost)

                obs = obs_next
                eps_ret, eps_cost, eps_len = eps_ret+rew, eps_cost+cost, eps_len+1

                if term or trun:
                    if term:
                        last_val, last_val_cost = 0.0, 0.0
                    else:
                        last_val, last_val_cost = self.evaluate(obs)
                    self.buffer.path_done(last_val, last_val_cost)

                    epoch_step += eps_len
                    self.num_eps += 1
                    eps_info = {'eps_ret': eps_ret, 'eps_cost': eps_cost, 'eps_len': eps_len}
                    self.logger.add(eps_info)
                    if self.use_tb:
                        self._to_tb(eps_info)
                    break
        return self.buffer.get(), epoch_step

    def evaluate(self, obs: np.ndarray) -> Union[float, float]:
        with torch.no_grad():
            obs_tensor = obs2ten(obs, self.device)
            v_obs = self.value(obs_tensor)
            v_cost_obs = self.value_cost(obs_tensor)
        return ten2arr(v_obs)[0], ten2arr(v_cost_obs)[0]
    
    def update_policy(self, data: PPOLagBufferItem):
        # update penalty parameter
        cur_cost = self.logger.mean('eps_costs')
        loss_penalty = -self.penalty_param*(cur_cost - self.cost_limit)
        self.penalty_optimizer.zero_grad()
        loss_penalty.backward()
        self.penalty_optimizer.step()

        adv = data.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)
        adv_cost = data.advantages_cost
        adv_cost = (adv_cost - adv_cost.mean()) / (adv_cost.std() + 1e-10)
        for _ in range(self.num_updates):
            log_probs = self.policy(data.observations, data.actions).squeeze()
            ratio = torch.exp(log_probs - data.log_probs)
            # policy loss: reward
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_policy_rew = (-torch.min(surr1, surr2)).mean()
            # policy loss: cost
            loss_policy_cost = (ratio*adv_cost).mean()
            # full policy loss
            p = softplus(self.penalty_param).item()
            loss_policy = 1/(1+p) * (loss_policy_rew + p * loss_policy_cost)

            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            self.policy_optimizer.step()

    def update_value_func(self, data: PPOLagBufferItem):
        for _ in range(self.num_updates):
            loss_val = ((self.value(data.observations).squeeze() - data.reward_to_gos)**2).mean()
            self.value_optimizer.zero_grad()
            loss_val.backward()
            self.value_optimizer.step()

            loss_val_cost = ((self.value_cost(data.observations).squeeze() - data.cost_to_gos)**2).mean()
            self.value_cost_optimizer.zero_grad()
            loss_val_cost.backward()
            self.value_cost_optimizer.step()

    def _init_hyperparameters(self, **kwargs):
        super()._init_hyperparameters(**kwargs)
        self.penalty_lr = kwargs.get('penalty_lr', 5e-2)
        self.cost_limit = kwargs.get('cost_limit', 100)

    def _init_networks(self):
        super()._init_networks()
        self.value_cost = Value(self.env).to(self.device)
        self.penalty_param = torch.tensor(1.0, requires_grad=True, dtype=torch.float32, device=self.device)
        
        self.value_cost_optimizer = Adam(self.value_cost.parameters(),lr=self.value_lr)
        self.penalty_optimizer = Adam([self.penalty_param], lr=self.penalty_lr)