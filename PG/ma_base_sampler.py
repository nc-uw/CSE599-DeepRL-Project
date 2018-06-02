#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:46:46 2018

@author: nehachoudhary
"""
#include 'o' in paths
#check propogation of 'action'
import math
import numpy as np
import ma_tensor_utils as tensor_utils

def pr(b, N, x): 
    price = b*N - np.sum(x)
    return price
    
def rev(price, mc, x):
    revenue = price*np.array(x) - np.multiply(mc,x)
    return revenue

def do_rollout(N, T, L, policy):
    
    amax = 50
    amin= 10
    mmm = (amax-amin)/3.5
    ccc =  (amax-amin)/2
    
    #N=4
    #T=2
    #L=10
    #do something about 'variables'
    variables = [i for i in range(N)]
    L = min(L, 100)
    paths = []
    
    for ep in range(T):
        #print('ep',ep)
        o = np.zeros(N)
        observations = {} 
        inp_nn = {} 
        actions = {}        
        #agent_infos = []        
        p=0
        price = []
        rewards = {}
        agent_info = {}
        done = {}
        
        for i in range(N):
            observations[variables[i]] = np.array(0)
            inp_nn[variables[i]] = np.empty(0)
            actions[variables[i]] = np.empty(0)
            agent_info[variables[i]] = []
            rewards[variables[i]] = np.empty(0)
            done[variables[i]] = np.empty(0)
        l = 0

        while l < L:
            #print('l',l)
            action=[]
            for i in range(N):
                inp = np.append(o,p)
                #print ("network-input", inp)
                if l == 0:
                    inp_nn[variables[i]] = np.array(inp)
                else:
                    inp_nn[variables[i]] = np.vstack((inp_nn[variables[i]],np.array(inp)))
                yyy, info = policy[i].get_action(inp)
                a = yyy
                if yyy >= 1.:
                    yyy = 1-1e-6 
                elif yyy <= -1:
                    yyy = -1+1e-6 
                    
                #print ("network-output", yyy)                
                #a = np.arctanh(yyy)*mmm + ccc
                '''
                if a > amax:
                    a = amax
                elif a < amin:
                    a = amin
                '''
                #print ("action",a)
                action.append(a)
                actions[variables[i]] = np.append(actions[variables[i]],a)
                agent_info[i].append(info)
                        
            action = np.ravel(action)
            #print ('action', action)
            p = pr(45., N, action)
            price.append(p)
            reward = rev(p, 15, action)
            
            for i in range(N):
                rewards[variables[i]] = np.append(rewards[variables[i]],reward[i])
                if l < L-1:
                    observations[variables[i]] = np.append(observations[variables[i]],action[i])
                    done[variables[i]] = np.append(done[variables[i]],False)
                    done[variables[i]] = False
                else:
                    done[variables[i]] = np.append(done[variables[i]],True)
                    done[variables[i]] = True
            o = action
            l += 1
            
        for i in range(N):
            agent_info[i] = tensor_utils.stack_tensor_dict_list(agent_info[i])
        #what happens to terminated?
        path = dict(
            o=inp_nn,
            observations=observations,
            actions=actions,
            rewards=rewards,
            agent_info=agent_info,
            terminated=done
        )

        paths.append(path)

    return paths

def do_rollout_star(args_list):
    return do_rollout(*args_list)