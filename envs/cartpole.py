import numpy as np
import gymnasium as gym
import math
import matplotlib.pyplot as plt

from typing import Dict, Optional, Union
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

def reset_paras(model_paras: Dict, altered_paras: Dict) -> Dict:
    for key, val in altered_paras.items():
        print(f'{key}:   Model = {model_paras[key]:.2f}  |  Plant = {val:.2f}')
        model_paras[key] = val
    return model_paras

class CartPole(CartPoleEnv):
    def __init__(self, altered_paras={}, render_mode = None, max_steps=250, xp=1):
        super().__init__(render_mode)
        self.xp = xp
        self.max_steps = max_steps
        model_paras = {
            'gravity': 9.8, 'masscart': 1.0,  'masspole': 0.1,
            'length': 0.5,  'force_mag': 10., 'dt': 0.02
        }
        for key, val in reset_paras(model_paras, altered_paras).items():
            exec(f'self.{key}={val}')
        self.masstotal = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        obs_high = np.array([4.8, 3.1, 24*math.pi/360, 5.0], dtype=np.float32,)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1., high=1., dtype=np.float32)
        self.init_state = np.array([0., 0., 6*2*math.pi/360, 0], dtype=np.float32)

    def step(self, action):
        self.time_step += 1
        x, x_dot, theta, theta_dot = self.state

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        force = min(max(action.item(), -1.0), 1.0)
        temp = (force*self.force_mag + self.polemass_length*theta_dot**2*sintheta) / self.masstotal
        thetaacc = (self.gravity*sintheta - costheta*temp) / (self.length*(4.0/3.0 - self.masspole*costheta**2/self.masstotal))
        xacc = temp - self.polemass_length*thetaacc*costheta / self.masstotal
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(
            x < -4.8
            or x > 4.8
            or theta < -24*math.pi/360
            or theta > 24*math.pi/360
        )
        # reward = -1e4 if terminated else -1000*theta**2 - self.xp*max(0, abs(x)-0.25)
        reward = 0 if terminated else 1
        cost = 10*theta**2 + x**2
        truncated = False if self.time_step<self.max_steps else True
        if self.render_mode == "human":
            self.render()
        return np.copy(self.state), reward, terminated, truncated, {"state": self.state, "meas": self.state, "cost": cost}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.time_step=0
        self.state = np.copy(self.init_state)

        if self.render_mode == "human":
            self.render()
        return np.copy(self.state), {"state": self.state, "meas": self.state, "cost": 0}
    
    def get_plot_obs(self, eps_obs: np.ndarray):
        assert eps_obs.ndim > 1, "_get_plot_obs() deals with obs with shape (N, num_obs)"
        eps_obs[:,2] = eps_obs[:,2]/np.pi*180   
        return eps_obs[:,[0,2]]
    
    def plot_traj(self, observations, *actions):
        num_steps = len(observations)
        # plot observations
        t = np.arange(0, self.dt*(num_steps-1)+0.0001, self.dt)
        t_c = np.arange(0, self.dt*(num_steps-2)+0.0001, self.dt/100.0)
        ax0 = plt.subplot(2,2,(1,2))
        plt.plot(t, observations[:, 0], color='C0', label=f'x')
        plt.legend(loc="upper left")
        ax1 = ax0.twinx()
        plt.plot(t, observations[:, 2]/math.pi*180, color='C1', label=r'\theta')
        plt.legend(loc="upper right")

        def plot_act(ax, act):
            act_0 = act[(t_c//self.dt).astype(int),0]
            plt.plot(t_c, act_0, f'C2', label="Force")
            plt.legend(loc="upper left")

        if len(actions) == 1:
            ax = plt.subplot(2,2,(3, 4))
            plot_act(ax, actions[0])
        else:
            ax = plt.subplot(2,2,3)
            plot_act(ax, actions[0])
            ax = plt.subplot(2,2,4)
            plot_act(ax, actions[1])
        plt.show()