#%reset
import numpy as np
import matplotlib.pyplot as plt
from ma_gaussian_mlp import MLP
from ma_linear_baseline import LinearBaseline
from ma_batch_reinforce import BatchREINFORCE
from ma_train_agent import train_agent

class MultiAgentNetwork:
    def __init__(self, N, seed, reg_coeff=1e-5):
        self.N = N
        self.variables = [i for i in range(self.N)]
        self.policy = {}
        for i in range(self.N):
            self.policy[self.variables[i]] = MLP(self.N, hidden_sizes=(128, 128), seed=SEED)


N = 4
L = 1e2
SEED = 123
baseline = LinearBaseline(N)
MAN= MultiAgentNetwork(N=4, seed=SEED)
policy = MAN.policy
agent = BatchREINFORCE(N, policy, baseline, learn_rate=0.5, seed=SEED, save_logs=True)
jobname = 'sim1'
    
mw_action_all, mw_reward_all, mw_price_all = train_agent(N, L, agent, 
seed=SEED, niter=25, gamma=0.0, gae_lambda=None, sample_mode='trajectories', num_traj=10, evaluation_rollouts=10)

returns_poster = {}
actions_poster = {}
price_poster = {}
variables = [i for i in range(N)]
for i in range(N):
    returns_poster[variables[i]] = np.empty(0)
    actions_poster[variables[i]] = np.empty(0)
    price_poster[variables[i]] = np.empty(0)


for i in range(N):
    for stats in stats_all:
       returns_poster[variables[i]] = np.append(returns_poster[variables[i]], stats[0][variables[i]])

    
for i in range(N):
    for p in eval_paths_all:
        for q in p:
            actions_poster[variables[i]] = np.append(actions_poster[variables[i]], np.mean(q['actions'][variables[i]]))
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
    
