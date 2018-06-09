#include 'o' in paths
#check propogation of 'action'
import numpy as np
import ma_tensor_utils as tensor_utils

def pr(b, N, x): 
    price = b*N - np.sum(x)
    return price
    
def rev(price, mc, x):
    revenue = price*np.array(x) - np.multiply(mc,x)
    return revenue

def do_rollout(N, T, L, policy):
    
    b = 45
    mc = 25
    
    amax = 40
    amin= 10
    m_slope = (amax-amin)/2
    c_intercept = (amax+amin)/2
    
    rmax= (pr(b, N, [amin]*(N-1)+[amax]) - mc) * amax
    rmin = (pr(b, N, [amax]*(N-1)+[amin]) - mc) * amin
    
    m2_slope = (rmax-rmin)*2
    c2_intercept = (rmax+rmin)/2
    
    pmax= pr(b, N, [amin]*N)
    pmin = pr(b, N, [amax]*N)
    
    m3_slope = (pmax-pmin)*2
    c3_intercept = (pmax+pmin)/2
    
    #N=4
    #T=2
    #L=10
    #do something about 'variables'
    variables = [i for i in range(N)]
    L = min(L, 1000)
    paths = []
    
    for ep in range(T):
        #print('ep',ep)
        o = np.zeros(N)
        observations = {} 
        inp_nn = {} 
        actions = {}        
        mw_actions = {}        
        #agent_infos = []        
        p=0
        price = []
        mw_price=[]
        rewards = {}
        mw_rewards = {}
        agent_info = {}
        done = {}
        
        for i in range(N):
            observations[variables[i]] = np.array(0)
            inp_nn[variables[i]] = np.empty(0)
            actions[variables[i]] = np.empty(0)
            mw_actions[variables[i]] = np.empty(0)
            agent_info[variables[i]] = []
            rewards[variables[i]] = np.empty(0)
            mw_rewards[variables[i]] = np.empty(0)
            done[variables[i]] = np.empty(0)
        l = 0

        while l < L:
            #print('l',l)
            mw_action=[]
            action = []
            for i in range(N):
                inp = np.append(o,p)
                if l == 0:
                    inp_nn[variables[i]] = np.array(inp)
                else:
                    inp_nn[variables[i]] = np.vstack((inp_nn[variables[i]],np.array(inp)))
                a, info = policy[i].get_action(inp)
                
                
                if a >= 1.:
                    a = [1-1e-6]
                elif a <= -1:
                    a = [-1+1e-6]
                 
                #print ("network-output", yyy)                
                
                mw_a = np.array(a)*m_slope + c_intercept
                
                if mw_a > amax:
                    mw_a = [amax]
                elif mw_a < amin:
                    mw_a = [amin]
                
                #print ('a',a)
                #print ('x',x)
                mw_action.append(mw_a) #mw_action for all agents
                action.append(a)
                actions[variables[i]] = np.append(actions[variables[i]],a)
                mw_actions[variables[i]] = np.append(actions[variables[i]],mw_a)
                agent_info[i].append(info)
                        
            mw_action = np.ravel(mw_action)
            action = np.ravel(action)
            #print ('mw_action', mw_action)
            #print ('next_o', next_o)
            
            mw_p = pr(b, N, mw_action)
            mw_price.append(mw_p)
            p = (np.array(mw_p) - c3_intercept)/m3_slope
            
            mw_reward = rev(mw_p, mc, mw_action)
            reward = (np.array(mw_reward) - c2_intercept)/m2_slope
            #scale
            
            for i in range(N):
                rewards[variables[i]] = np.append(rewards[variables[i]],reward[i])
                mw_reward[variables[i]] = np.append(mw_rewards[variables[i]],mw_reward[i])
                if l < L-1:
                    observations[variables[i]] = np.append(observations[variables[i]],action[i])
                    #done[variables[i]] = np.append(done[variables[i]],False)
                    done[variables[i]] = False
                else:
                    #done[variables[i]] = np.append(done[variables[i]],True)
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
            mw_actions=mw_actions,
            mw_rewards=mw_rewards,
            agent_info=agent_info,
            terminated=done
        )

        paths.append(path)

    return paths

def do_rollout_star(args_list):
    return do_rollout(*args_list)