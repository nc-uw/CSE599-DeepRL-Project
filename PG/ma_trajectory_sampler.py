import numpy as np
import copy
import multiprocessing as mp
import time as timer
import ma_base_sampler as base_sampler
import ma_evaluation_sampler as eval_sampler

def sample_paths(N, T, L, policy, mode='sample'):
    if mode == 'sample':
        return base_sampler.do_rollout(N, T, L, policy)
    elif mode == 'evaluation':
        return eval_sampler.do_evaluation_rollout(N, T, L, policy)
    else:
        print("Mode has to be either 'sample' for training time or 'evaluation' for test time performance")

'''
def sample_paths_parallel(N, T, L, policy):
    return base_sampler.do_rollout(N, T, L, policy)
'''
