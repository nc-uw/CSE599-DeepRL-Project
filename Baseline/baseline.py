import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import math
import sys

def inverse_demand(d, saturation):
    p = saturation - d
    return p

total_iters = 5000
gamma = 0
epsilon = 0.1
alpha = 0.1


if __name__ == '__main__':
    N = int(sys.argv[1])
    mc = int(sys.argv[2])
    min_quantity = int(sys.argv[3])
    max_quantity = int(sys.argv[4])

    P = []
    bandits = dict()
    for i in range(N):
        name = 'actions_' + chr(ord('a') + i)
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
        #d = math.sqrt(bids[0] ** 2 + bids[1] ** 2 + bids[2] ** 2 + bids[3] ** 2)
        saturation = 25 * N
        p = inverse_demand(d, saturation)
        print('price', p)

        for name in bandits:
            bandit = bandits[name]
            bid = bandit['bids'][-1]
            max_bid = bandit['max_bids'][-1]
            reward = (p - mc) * bid
            if i % 100 == 0:
                bandit['reward'].append(reward)
            bandit['value_function'][bid] = (1 - alpha) * bandit['value_function'][bid] + alpha * (reward + gamma * max_bid)

        epsilon = epsilon - epsilon * (float((i * 0.005)) / float(total_iters))
        print('epsilon', epsilon)
        P.append(p)

    for name in bandits:
        print bandits[name]['value_function']

    axis = np.arange(0, total_iters, 100)
    plt.figure()
    plt.title('Reward trajectory for 4 n-armed bandits \nExploration: decaying, Alpha: 0.2, Discount Factor: 0.9')
    for name in bandits:
        plt.plot(axis, bandits[name]['reward'], label=name)
    plt.ylabel('Net Revenue')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Bid trajectory for 4 n-armed bandits \nExploration: 0.975-greedy, Alpha: 0.2, Discount Factor: 0.9')
    for name in bandits:
        bids = bandits[name]['bids']
        bids = [bids[i] for i in axis]
        plt.plot(axis, bids, label=name)
    plt.ylabel('Qty Bid')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Price')
    plt.plot(P, 'k--', label='Cournot Price')
    plt.ylabel('Price')
    plt.xlabel('Iterations')
    plt.legend()
    plt.show()