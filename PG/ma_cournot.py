#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:57:09 2018

@author: nehachoudhary
"""
#%reset
from ma_gaussian_mlp import MLP
from ma_linear_baseline import LinearBaseline
from ma_batch_reinforce import BatchREINFORCE
from ma_train_agent import train_agent

class MultiAgentNetwork:
    def __init__(self, N, reg_coeff=1e-5):
        self.N = N
        self.variables = [i for i in range(self.N)]
        self.policy = {}
        for i in range(self.N):
            self.policy[self.variables[i]] = MLP(self.N, hidden_sizes=(32,32), seed=888)

SEED = 500
N = 4
L = 1e2
baseline = LinearBaseline(N)
MAN= MultiAgentNetwork(4)
policy = MAN.policy
agent = BatchREINFORCE(N, policy, baseline, learn_rate=0.1, seed=SEED)

paths = train_agent(N, L, agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.9,
            gae_lambda=0.5,
            sample_mode='trajectories',
            num_traj=10,
            evaluation_rollouts=10)