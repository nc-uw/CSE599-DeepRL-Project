#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:34:13 2018

@author: nehachoudhary
"""

import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

#samplers
import ma_trajectory_sampler as trajectory_sampler
import ma_process_samples as process_samples

class BatchREINFORCE:
    def __init__(self, N, policy, baseline,
                 learn_rate=0.01,
                 seed=None):

        self.N = N
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.running_score_init = None
        self.variables = [i for i in range(self.N)]

    def CPI_surrogate(self, i, observations, actions, advantages):        
        advantages = advantages / (np.max(advantages) + 1e-8)
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = self.policy[i].old_dist_info(i, observations, actions)
        #old_dist_info = [LL, mean, policy[i].old_log_std]
        new_dist_info = self.policy[i].new_dist_info(i, observations, actions)    
        #new_dist_info = [LL, mean, policy[i].log_std]
        LR = self.policy[i].likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        print ('\n\nsurr', surr, 'LR', LR, 'LR_compute', torch.exp(new_dist_info[0] - old_dist_info[0]), 'new_dist_info', new_dist_info[0], 'old_dist_info', old_dist_info[0])
        return surr

    def kl_old_new(self, i, observations, actions):
        old_dist_info = self.policy[i].old_dist_info(i, observations, actions)
        new_dist_info = self.policy[i].new_dist_info(i, observations, actions)
        mean_kl = self.policy[i].mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, i, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(i, observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy[i].trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad
        
    '''
    def CPI_surrogate(i, observations, actions, advantages):        
        advantages = advantages / (np.max(advantages) + 1e-8)
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        old_dist_info = policy[i].old_dist_info(i, observations, actions)
        #old_dist_info = [LL, mean, policy[i].old_log_std]
        new_dist_info = policy[i].new_dist_info(i, observations, actions)    
        #new_dist_info = [LL, mean, policy[i].log_std]
        LR = policy[i].likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR*adv_var)
        return surr

    def kl_old_new(i, observations, actions):
        old_dist_info = policy[i].old_dist_info(i, observations, actions)
        new_dist_info = policy[i].new_dist_info(i, observations, actions)
        mean_kl = policy[i].mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(i, observations, actions, advantages):
        cpi_surr = CPI_surrogate(i, observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, policy[i].trainable_params)
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad
    '''
    
    # ----------------------------------------------------------
    def train_step(self, T,
                   sample_mode='trajectories',
                   L=1e2,
                   gamma=0.9,
                   gae_lambda=0.98):

        if sample_mode is not 'trajectories' and sample_mode is not 'samples':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        #edit for seed 
        #ensure N, T, L are fine
        if sample_mode is 'trajectories':
            paths = trajectory_sampler.sample_paths(self.N, T, L, self.policy)
        '''
        elif sample_mode is 'samples':
            paths = batch_sampler.sample_paths(N, self.policy, T, env_name=env_name,
                                               pegasus_seed=self.seed, num_cpu=num_cpu)
        '''
        #paths = trajectory_sampler.sample_paths_parallel(N, T, L, policy)
        ##whatis this below
        self.seed = self.seed + T if self.seed is not None else self.seed
        # compute returns
        paths = process_samples.compute_returns(self.N, paths, gamma)
        # compute advantages
        paths = process_samples.compute_advantages(self.N, paths, self.baseline, gamma, gae_lambda)
        #paths = process_samples.compute_advantages(N, paths, baseline, gamma=0.9, gae_lambda=0.98, normalize=False)
        #print ("\n\n\npaths", paths)
        # train from paths
        eval_statistics, optimization_stats = self.train_from_paths(paths)
        eval_statistics.append(T)
        # fit baseline
        error_before, error_after = self.baseline.fit(paths, return_errors=True)
        return eval_statistics, optimization_stats, paths

    # ----------------------------------------------------------
    def train_from_paths(self, paths):
        
        # Concatenate from all the trajectories
        observations = {}  
        actions = {}
        advantages = {}
        path_returns = {}
        mean_return = {}
        std_return = {}
        min_return = {}
        max_return = {}
        running_score = {}
        for i in range(self.N):
            running_score[self.variables[i]] = self.running_score_init
            #running_score[variables[i]] = running_score_init
            
        for i in range(self.N):
            observations[self.variables[i]] = np.concatenate([path["o"][self.variables[i]] for path in paths])
            actions[self.variables[i]] = np.concatenate([path["actions"][self.variables[i]] for path in paths])
            advantages[self.variables[i]] = np.concatenate([path["advantages"][self.variables[i]] for path in paths])
            # Advantage whitening
            advantages[self.variables[i]] = (advantages[self.variables[i]] - np.mean(advantages[self.variables[i]])) / (np.std(advantages[self.variables[i]]) + 1e-6)

            # cache return distributions for the paths
            path_returns[self.variables[i]] = [sum(p["rewards"][self.variables[i]]) for p in paths]
            mean_return[self.variables[i]] = np.mean(path_returns[self.variables[i]])
            std_return[self.variables[i]] = np.std(path_returns[self.variables[i]])
            min_return[self.variables[i]] = np.amin(path_returns[self.variables[i]])
            max_return[self.variables[i]] = np.amax(path_returns[self.variables[i]])
            #base_stats[self.variables[i]] = [mean_return[self.variables[i]], std_return[self.variables[i]], min_return[self.variables[i]], max_return[self.variables[i]]]
            running_score[self.variables[i]] = mean_return[self.variables[i]] if running_score[self.variables[i]] is None else \
                                 0.9*running_score[self.variables[i]] + 0.1*mean_return[self.variables[i]]  # approx avg of last 10 iters
            '''
            running_score[variables[i]] = mean_return[variables[i]] if running_score[variables[i]] is None else \
                                 0.9*running_score[variables[i]] + 0.1*mean_return[variables[i]]  # approx avg of last 10 iters
            '''
        base_stats = [mean_return, running_score]
        # Optimization algorithm
        # --------------------------
        '''
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        curr_params = self.policy.get_param_values()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        '''
        surr_before = {}
        curr_params = {}
        vpg_grad = {}
        new_params = {}
        new_surr = {}
        kl_dist = {}
        surr_improvement = {}
        for i in range(self.N):
            surr_before[self.variables[i]]= self.CPI_surrogate(i, observations[self.variables[i]], actions[self.variables[i]], advantages[self.variables[i]]).data.numpy().ravel()[0]            
            #surr_before[variables[i]]= CPI_surrogate(i, observations[variables[i]], actions[variables[i]], advantages[variables[i]]).data.numpy().ravel()[0]            
            curr_params[self.variables[i]] = self.policy[i].get_param_values()
            #curr_params[variables[i]] = policy[i].get_param_values()
            vpg_grad[self.variables[i]] = self.flat_vpg(i, observations[self.variables[i]], actions[self.variables[i]], advantages[self.variables[i]])
            #vpg_grad[variables[i]] = flat_vpg(i, observations[variables[i]], actions[variables[i]], advantages[variables[i]])
            new_params[self.variables[i]], new_surr[self.variables[i]], kl_dist[self.variables[i]] = self.simple_gradient_update(i, curr_params[self.variables[i]], vpg_grad[self.variables[i]], self.alpha, observations[self.variables[i]], actions[self.variables[i]], advantages[self.variables[i]])
            #new_params[variables[i]], new_surr[variables[i]], kl_dist[variables[i]] = simple_gradient_update(i, curr_params[variables[i]], vpg_grad[variables[i]], alpha, observations[variables[i]], actions[variables[i]], advantages[variables[i]])
            self.policy[i].set_param_values(new_params[self.variables[i]], set_new=True, set_old=True)
            #policy[i].set_param_values(new_params[variables[i]], set_new=True, set_old=True)
            surr_improvement[self.variables[i]] = new_surr[self.variables[i]] - surr_before[self.variables[i]]
            #opt_pg_stats[self.variables[i]] = [surr_before[self.variables[i]], new_surr[self.variables[i]], kl_dist[self.variables[i]]]
        
        opt_pg_stats = [surr_before, new_surr, kl_dist]
        return base_stats, opt_pg_stats
    
    '''
    def simple_gradient_update(self, i, curr_params, search_direction, step_size,
                               observations, actions, advantages):
        # This function takes in the current parameters, a search direction, and a step size
        # and computes the new_params =  curr_params + step_size * search_direction.
        # It also computes the CPI surrogate at the new parameter values.
        # This function also computes KL(pi_new || pi_old) as discussed in the class,
        # where pi_old = policy with current parameters (i.e. before any update),
        # and pi_new = policy with parameters equal to the new_params as described above.
        # The function DOES NOT set the parameters to the new_params -- this has to be
        # done explicitly outside this function.

        new_params = curr_params + step_size*search_direction
        self.policy[i].set_param_values(new_params, set_new=True, set_old=False)
        new_surr = self.CPI_surrogate(i, observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(i, observations, actions).data.numpy().ravel()[0]
        self.policy[i].set_param_values(curr_params, set_new=True, set_old=True)
        return new_params, new_surr, kl_dist
    '''
    
    def simple_gradient_update(self, i, curr_params, search_direction, step_size, observations, actions, advantages):
        # This function takes in the current parameters, a search direction, and a step size
        # and computes the new_params =  curr_params + step_size * search_direction.
        # It also computes the CPI surrogate at the new parameter values.
        # This function also computes KL(pi_new || pi_old) as discussed in the class,
        # where pi_old = policy with current parameters (i.e. before any update),
        # and pi_new = policy with parameters equal to the new_params as described above.
        # The function DOES NOT set the parameters to the new_params -- this has to be
        # done explicitly outside this function.

        new_params = curr_params + step_size*search_direction
        self.policy[i].set_param_values(new_params, set_new=True, set_old=False)
        new_surr = self.CPI_surrogate(i, observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(i, observations, actions).data.numpy().ravel()[0]
        self.policy[i].set_param_values(curr_params, set_new=True, set_old=True)
        return new_params, new_surr, kl_dist