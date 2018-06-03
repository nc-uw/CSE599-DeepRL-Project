#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:51:58 2018

@author: nehachoudhary
"""

from ma_trajectory_sampler import sample_paths
import numpy as np
import os
import copy

def train_agent(N, L, agent,
                seed = 0,
                niter = 100,
                gamma = 0.9,
                gae_lambda = None,
                sample_mode = 'trajectories',
                num_traj = 50,
                evaluation_rollouts = None,
                ):

    np.random.seed(seed)
    variables = [i for i in range(N)]
    best_policy = {}
    train_curve = {}
    best_perf = {}
    best_perf_init = -1e8
    mean_pol_perf = {}
    mean_pol_perf_init = 0.0
    T = num_traj
            
    for i in range(N):
        best_policy[variables[i]] = copy.deepcopy(agent.policy[variables[i]])
        best_perf[variables[i]] = best_perf_init
        train_curve[variables[i]] = best_perf_init*np.ones(niter)
        mean_pol_perf[variables[i]] = mean_pol_perf_init

    for i in range(niter):
        print("......................................................................................")
        print("ITERATION : %i " % i)
        for j in range(N):
            if train_curve[variables[j]][i-1] > best_perf[variables[j]]:
                best_policy[variables[j]] = copy.deepcopy(agent.policy[variables[j]])
                best_perf[variables[j]] = train_curve[variables[j]][i-1]
        args = dict(T=T, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda)
        stats, paths = agent.train_step(**args)
        print ("\n\n\nstats", stats)
        for j in range(N):
            train_curve[variables[j]][i] = stats[variables[j]][0]
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            eval_paths = sample_paths(N, T, L, policy=agent.policy, mode='evaluation')
            for j in range(N):        
                mean_pol_perf[variables[j]] = np.mean([np.sum(path['rewards'][variables[j]]) for path in eval_paths])
        print ('paths', paths)
    return paths
