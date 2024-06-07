import torch
import scipy
import numpy as np
from typing import Dict


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """Compute discounted cumulative sums
    Input:
        x = [x0, x1, x2]
    Output:
        [x0 + d * x1 + d^2 * x2, x1 + d * x2, x2]"""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def gae_lambda(rews: np.ndarray, vals: np.ndarray, gamma: float, lam: float=0.97) -> np.ndarray:
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    return discount_cumsum(deltas, gamma*lam)

def obs2ten(x: np.ndarray, target_device: torch.device) -> torch.Tensor:
    return torch.Tensor(x).unsqueeze(0).to(target_device)

def ten2arr(x: torch.Tensor) -> np.ndarray:
    return x.squeeze(0).detach().cpu().numpy()

class Logger:
    def __init__(self):
        self.eps_rets = []
        self.eps_costs = []
        self.eps_lens = []
    
    def add(self, eps_info: Dict):
        self.eps_rets.append(eps_info['eps_ret'])
        self.eps_costs.append(eps_info['eps_cost'])
        self.eps_lens.append(eps_info['eps_len'])
    
    def mean(self, var: str) -> float:
        if hasattr(self, var):
            values = getattr(self, var)
            return sum(values) / len(values) if values else 0
        else:
            raise AttributeError(f"'Logger' object has no attribute '{var}'")