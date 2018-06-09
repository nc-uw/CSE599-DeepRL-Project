from ma_trajectory_sampler import sample_paths
import numpy as np
import os
import copy

def train_agent(N, L, agent,
                seed = 0,
                niter = 100,
                gamma = 0.998,
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
    
    mean_pol_perf_all = []
    stats_all = []
    optimization_stats_all = []
    paths_all = []
    eval_paths_all = []
    train_curve_all = []
    mw_action_all = []
    mw_reward_all = []
    mw_price_all = []

    for i in range(N):
        best_policy[variables[i]] = copy.deepcopy(agent.policy[variables[i]])
        best_perf[variables[i]] = best_perf_init
        train_curve[variables[i]] = best_perf_init*np.ones(niter)
        mean_pol_perf[variables[i]] = mean_pol_perf_init

    for i in range(niter):
        print("......................................................................................")
        print("\n\nITERATION : %i " % i)
        for j in range(N):
            if train_curve[variables[j]][i-1] > best_perf[variables[j]]:
                best_policy[variables[j]] = copy.deepcopy(agent.policy[variables[j]])
                best_perf[variables[j]] = train_curve[variables[j]][i-1]
        args = dict(T=T, sample_mode=sample_mode, gamma=gamma, gae_lambda=gae_lambda)
        stats, optimization_stats, paths, bl_error = agent.train_step(**args)
        print ("\nstats", stats)
        print ("\nopt pg stats", optimization_stats)
        print ("\nopt bl_error", bl_error)
        for j in range(N):
            train_curve[variables[j]][i] = stats[0][variables[j]]
        if evaluation_rollouts is not None and evaluation_rollouts > 0:
            print("Performing evaluation rollouts ........")
            mw_action, mw_reward, mw_price, eval_paths = sample_paths(N, T, L, policy=agent.policy, mode='evaluation')
            for j in range(N):        
                mean_pol_perf[variables[j]] = np.mean([np.sum(path['rewards'][variables[j]]) for path in eval_paths])
        print ('mean_pol_perf', mean_pol_perf)
        #print ('train_curve', train_curve)
        mean_pol_perf_all.append(mean_pol_perf)
        mean_pol_perf_all.append(mean_pol_perf)
        stats_all.append(stats)
        optimization_stats_all.append(optimization_stats)
        paths_all.append(paths)
        eval_paths_all.append(eval_paths)
        train_curve_all.append(train_curve)
        mw_action_all.append(mw_action)
        mw_reward_all.append(mw_reward)
        mw_price_all.append(mw_price)
    #return stats_all, optimization_stats_all, paths_all, eval_paths_all, mean_pol_perf_all, train_curve_all
    return mw_action_all, mw_reward_all, mw_price_all