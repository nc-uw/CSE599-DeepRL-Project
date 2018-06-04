import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import math
import sys

def inverse_demand(d, saturation):
    p = saturation - d
    return p

total_iters = 5000
gamma = 0.9
epsilon = 1
alpha = 0.1


if __name__ == '__main__':
    N = int(sys.argv[1])
    mc = int(sys.argv[2])
    min_quantity = int(sys.argv[3])
    max_quantity = int(sys.argv[4])
    saturation = int(sys.argv[5])

    P = []
    bandits = dict()
    for i in range(N):
        name = 'Agent_' + chr(ord('A') + i)
        bandits[name] = dict()
        bandits[name]['actions'] = np.arange(min_quantity, max_quantity, 1.0)
        bandits[name]['value_function'] = {action: 0 for action in bandits[name]['actions']}
        bandits[name]['reward'] = []
        bandits[name]['bids'] = []
        bandits[name]['max_bids'] = []

    for i in range(total_iters):
        print('iter', i)

        bids = []
        for name in bandits:
            bandit = bandits[name]
            value_function = bandit['value_function']
            max_bid = list(value_function.keys())[list(value_function.values()).index(max(value_function.values()))]
            if rdm.uniform(0, 1) <= (1 - epsilon):
                bid = max_bid
                print("max bid", bid)
            else:
                bid = rdm.choice(bandit['actions'])
                print("random bid", bid)
            bandit['bids'].append(bid)
            bandit['max_bids'].append(max_bid)
            bids.append(bid)

        d = sum(bids)
        # d = 0
        # for bid in bids:
        #     d += bid ** 2
        # d = np.sqrt(d)
        #d = math.sqrt(bids[0] ** 2 + bids[1] ** 2 + bids[2] ** 2 + bids[3] ** 2)
        p = inverse_demand(d, saturation)
        print('price', p)

        for name in bandits:
            bandit = bandits[name]
            bid = bandit['bids'][-1]
            max_bid = bandit['max_bids'][-1]
            reward = (p - mc) * bid
            bandit['reward'].append(reward)
            bandit['value_function'][bid] = (1 - alpha) * bandit['value_function'][bid] + alpha * (reward + gamma * max_bid)
            #bandit['value_function'][bid] = bandit['value_function'][bid] + alpha (reward - bandit['value_function'][bid])
        epsilon = max(epsilon - epsilon * (float((i * 0.005)) / float(total_iters)), 0.001)
        print('epsilon', epsilon)
        P.append(p)

    # for name in bandits:
    #     print bandits[name]['value_function']

    axis = np.arange(100, total_iters, 100)

    plt.figure()
    plt.title('Reward trajectory for ' + str(N) + ' n-armed bandits \nExploration: decaying, Alpha: ' + str(alpha) + ', Discount Factor: ' + str(gamma))
    for name in sorted(bandits.keys()):
        rewards = bandits[name]['reward']
        graph_reward = [0]
        avgs = [np.mean(rewards[i-100:i]) for i in axis]
        graph_reward += avgs
        axis = np.insert(axis, 0, 0)
        plt.plot(axis, graph_reward, label=name)
        axis = np.arange(100, total_iters, 100)
    plt.ylabel('Net Revenue')
    plt.xlabel('Iterations')
    plt.ylim(4000, 6500)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Bid trajectory for ' + str(N) + ' n-armed bandits \nExploration: decaying, Alpha: ' + str(alpha) + ', Discount Factor: ' + str(gamma))
    for name in sorted(bandits.keys()):
        bids = bandits[name]['bids']
        graph_bid = [bids[0]]
        avgs = [np.mean(bids[i-100:i]) for i in axis]
        graph_bid += avgs
        axis = np.insert(axis, 0, 0)
        plt.plot(axis, graph_bid, label=name)
        axis = np.arange(100, total_iters, 100)
    plt.ylabel('Qty Bid')
    plt.xlabel('Iterations')
    plt.ylim(5, 35)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Market Price')
    plt.plot(P, 'k--', label='Cournot Price')
    plt.ylabel('Price')
    plt.xlabel('Iterations')
    plt.legend()
    plt.ylim(0, 300)
    plt.show()