import numpy as np
import torch
from ppo.utils import gae_lambda, discount_cumsum


class PPOBufferItem:
    def __init__(self, device: torch.device, observations: np.ndarray, actions: np.ndarray, 
                 log_probs: np.ndarray, reward_to_gos: np.ndarray, advantages: np.ndarray):
        self.observations = torch.tensor(np.array(observations), dtype=torch.float32, device=device)
        self.actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        self.log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)
        self.reward_to_gos = torch.tensor(np.array(reward_to_gos), dtype=torch.float32, device=device)
        self.advantages = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)

class PPOLagBufferItem(PPOBufferItem):
    def __init__(self, device: torch.device, observations: np.ndarray, actions: np.ndarray, 
                 log_probs: np.ndarray, reward_to_gos: np.ndarray, advantages: np.ndarray, 
                 cost_to_gos: np.ndarray, advantages_cost: np.ndarray):
        super().__init__(device, observations, actions, log_probs, reward_to_gos, advantages)
        self.cost_to_gos = torch.tensor(np.array(cost_to_gos), dtype=torch.float32, device=device)
        self.advantages_cost = torch.tensor(np.array(advantages_cost), dtype=torch.float32, device=device)

class PPOBuffer:
    def __init__(self, device: torch.device, gamma: float=0.99) -> None:
        self.device = device
        self.gamma = gamma
        self._set_buffers()

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float):
        self._add_transition(obs, act, rew, val, logp)
        self.ptr += 1

    def path_done(self, last_val: float):
        """
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T)
        """
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(np.array(self.rew_buf, dtype=np.float32)[path_slice], last_val)
        vals = np.append(np.array(self.val_buf, dtype=np.float32)[path_slice], last_val)

        self.rtg_buf += discount_cumsum(rews, self.gamma)[:-1].tolist()
        self.adv_buf += gae_lambda(rews, vals, self.gamma).tolist()

        self.path_start_idx = self.ptr
    
    def get(self) -> PPOBufferItem:
        data = PPOBufferItem(self.device, self.obs_buf, self.act_buf, self.logp_buf, 
                             self.rtg_buf, self.adv_buf)
        self._set_buffers()
        return data

    def _add_transition(self, obs, act, rew, val, logp):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)
    
    def _set_buffers(self):
        self.ptr, self.path_start_idx = 0, 0
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.rtg_buf = []
        self.adv_buf = []

class PPOLagBuffer(PPOBuffer):
    def __init__(self, device: torch.device, gamma: float=0.99) -> None:
        self.device = device
        self.gamma = gamma
        self._set_buffers()
    
    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float, 
            cost: np.ndarray, val_cost: np.ndarray):
        self._add_transition(obs, act, rew, val, logp, cost, val_cost)
        self.ptr += 1

    def path_done(self, last_val: float, last_val_cost: float):
        """
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T)
        """
        path_slice = slice(self.path_start_idx, self.ptr)

        rews = np.append(np.array(self.rew_buf, dtype=np.float32)[path_slice], last_val)
        vals = np.append(np.array(self.val_buf, dtype=np.float32)[path_slice], last_val)
        costs = np.append(np.array(self.cost_buf, dtype=np.float32)[path_slice], last_val_cost)
        vals_cost = np.append(np.array(self.val_cost_buf, dtype=np.float32)[path_slice], last_val_cost)

        self.rtg_buf += discount_cumsum(rews, self.gamma)[:-1].tolist()
        self.adv_buf += gae_lambda(rews, vals, self.gamma).tolist()
        self.ctg_buf += discount_cumsum(costs, self.gamma)[:-1].tolist()
        self.adv_cost_buf += gae_lambda(costs, vals_cost, self.gamma).tolist()

        self.path_start_idx = self.ptr

    def get(self) -> PPOLagBufferItem:
        data = PPOLagBufferItem(self.device, self.obs_buf, self.act_buf, self.logp_buf, 
                             self.rtg_buf, self.adv_buf, self.cost_buf, self.adv_cost_buf)
        self._set_buffers()
        return data
    
    def _add_transition(self, obs, act, rew, val, logp, cost, val_cost):
        super()._add_transition(obs, act, rew, val, logp)
        self.cost_buf.append(cost)
        self.val_cost_buf.append(val_cost)
    
    def _set_buffers(self):
        super()._set_buffers()
        self.cost_buf = []
        self.val_cost_buf = []
        self.ctg_buf = []
        self.adv_cost_buf = []