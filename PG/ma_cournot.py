#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:57:09 2018

@author: nehachoudhary
"""
#%reset
import numpy as np
import matplotlib.pyplot as plt
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

SEED = 888
N = 4
L = 1e2
baseline = LinearBaseline(N)
MAN= MultiAgentNetwork(4)
policy = MAN.policy
agent = BatchREINFORCE(N, policy, baseline, learn_rate=0.5, seed=SEED)
    
stats_all, optimization_stats_all, paths_all, eval_paths_all, mean_pol_perf_all, train_curve_all = train_agent(N, L, agent=agent,
                seed=SEED,
                niter=25,
                gamma=0.3,
                gae_lambda=None,
                sample_mode='trajectories',
                num_traj=5,
                evaluation_rollouts=5)

#returns_poster = {}
#actions_poster = {}
price_poster = {}
variables = [i for i in range(N)]
for i in range(N):
    #returns_poster[variables[i]] = np.empty(0)
    #actions_poster[variables[i]] = np.empty(0)
    price_poster[variables[i]] = np.empty(0)

'''
for i in range(N):
    for stats in stats_all:
       returns_poster[variables[i]] = np.append(returns_poster[variables[i]], stats[0][variables[i]])
'''
    
for i in range(N):
    for p in paths_all:
        for q in p:
            #actions_poster[variables[i]] = np.append(actions_poster[variables[i]], np.mean(q['actions'][variables[i]]))
            for r in q['o'][variables[i]][-1]:
                price_poster[variables[i]] = np.append(price_poster[variables[i]], np.mean(r))

plt.figure()
plt.title('Reward trajectory for 4 n-armed bandits \nVanilla Policy Grad, Alpha: 0.1, Discount Factor: 0.9')
plt.plot(returns_poster[0]/100,'r--', label = 'Agent A')
plt.plot(returns_poster[1]/100, 'b--', label = 'Agent B')
plt.plot(returns_poster[2]/100, 'g--', label = 'Agent C')
plt.plot(returns_poster[3]/100, 'y--', label = 'Agent D')
plt.ylabel('Net Revenue')
plt.xlabel('Iterations')
plt.legend()
plt.show()

plt.figure()
plt.title('Bid trajectory for 4 n-armed bandits \nVanilla Policy Grad, Alpha: 0.1, Discount Factor: 0.9')
plt.plot(actions_poster[0],'r--', label = 'Agent A')
plt.plot(actions_poster[1], 'b--', label = 'Agent B')
plt.plot(actions_poster[2], 'g--', label = 'Agent C')
plt.plot(actions_poster[3], 'y--', label = 'Agent D')
plt.ylabel('Qty Bid')
plt.xlabel('Iterations')
plt.legend()
plt.show()

plt.figure()
plt.title('Avg. Price')
plt.plot(price_poster[0],'k--', label = 'Cournot Price')
plt.ylabel('Price')
plt.xlabel('Iterations')
plt.legend()
plt.show()
    
