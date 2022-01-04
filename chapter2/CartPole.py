# -*- coding: utf-8 -*-
'''
@Author: tanbo
@Time: 2021/12/26 9:29 下午
@File: 。..py
'''
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import loguru
logger = loguru.logger
gamma = 0.99


class Pi(nn.Module):
    def __init__(self, id_dim, out_dim):
        super(Pi, self).__init__()
        layers = [nn.Linear(id_dim, 64),
                  nn.ReLU(),
                  nn.Linear(64, out_dim)]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x) # 输出维度大于1
        pd = Categorical(logits=pdparam) # action个数与输出维度一致？
        action = pd.sample() # pi(a|s)
        # todo：action如何采样才合理？
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()


def train(pi, optimizer):
    """
    根据累积的state对应的rewards计算loss
    todo:思考这个reward是如何设计的呢？
    :param pi:
    :param optimizer:
    :return:
    """
    # 强化学习下的梯度下降
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma*future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets # 强化学习中的loss是什么呢？实际中的reward如何设计？
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    env = gym.make("CartPole-v0")
    in_dim = env.observation_space.shape[0] # 4
    out_dim = env.action_space.n # 2
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(300):
        state = env.reset()
        for t in range(200): # cartpole max timestep is 200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            # logger.info(f" reward:{reward}")
            env.render()
            if done:
                break
        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195
        pi.onpolicy_reset() # 训练后清除内存
        logger.info(f"【episode】:{epi}; 【loss】:{loss}; 【total_reward】:{total_reward}; 【solved】:{solved}")


if __name__ == "__main__":
    main()