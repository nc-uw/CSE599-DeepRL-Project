import numpy as np

def compute_returns(N, paths, gamma):
    #gamma = 0.9
    variables = [i for i in range(N)]
    for i in range(len(paths)):
        returns = {}
        for j in range(N):
            returns[variables[j]] = discount_sum(paths[i]["rewards"][variables[j]], gamma)
        paths[i].update(dict(returns=returns))
    return paths

def compute_advantages(N, paths, baseline, gamma, gae_lambda=None, normalize=False):
    #gamma = 0.9
    #gae_lambda = None
    variables = [i for i in range(N)]
    # compute and store returns, advantages, and baseline 
    # standard mode
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        for i in range(len(paths)):
            base_line = baseline.predict(paths[i])
            advantages = {}
            for j in range(N):
                advantages[variables[j]] = paths[i]["returns"][variables[j]] - base_line[variables[j]]
            paths[i].update(dict(baseline=base_line))
            paths[i].update(dict(advantages=advantages))          
            
        if normalize:
            mean_adv = {}
            std_adv = {}
            for i in range(N):
                alladv = np.concatenate([path["advantages"][variables[i]] for path in paths])
                mean_adv[variables[i]] = alladv.mean()
                std_adv[variables[i]] = alladv.std()
            
            for i in range(len(paths)):
                adv_norm = {}
                for j in range(N):                
                    adv_norm[variables[j]] = (paths[i]["advantages"][j]-mean_adv[variables[j]])/(std_adv[variables[j]]+1e-8)
                paths[i].update(dict(advantages=adv_norm))
    # GAE mode
    else:
        for i in range(len(paths)):
            b = paths[i]["baseline"] = baseline.predict(paths[i])
            b1 = {}
            td_deltas = {}
            advantages = {}
            for j in range(N):
                if b[variables[j]].ndim == 1:
                    b1[variables[j]] = np.append(paths[i]["baseline"][variables[j]], 0.0 if paths[i]["terminated"][variables[j]] else b[variables[j]][-1])
                else:
                    b1[variables[j]] = np.vstack((b[variables[j]], np.zeros(b[variables[j]].shape[1]) if paths[i]["terminated"][variables[j]] else b[variables[j]][-1]))
                td_deltas[variables[j]] = paths[i]["rewards"][variables[j]] + gamma*b1[variables[j]][1:] - b1[variables[j]][:-1]
                #what_is_this_below?!
                advantages[variables[j]] = discount_sum(td_deltas[variables[j]], gamma*gae_lambda)
            paths[i].update(dict(advantages=advantages))            
        
        if normalize:
            mean_adv = {}
            std_adv = {}
            for i in range(N):
                alladv = np.concatenate([path["advantages"][variables[i]] for path in paths])
                mean_adv[variables[i]] = alladv.mean()
                std_adv[variables[i]] = alladv.std()
            
            for i in range(len(paths)):
                adv_norm = {}
                for j in range(N):                
                    adv_norm[variables[j]] = (paths[i]["advantages"][j]-mean_adv[variables[j]])/(std_adv[variables[j]]+1e-8)
                paths[i].update(dict(advantages=adv_norm))
    return paths

def discount_sum(x, gamma, terminal=0.0):
    y = []
    run_sum = terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])