"""
Created on Tue May 15 19:13:19 2018

@author: nc57
"""

import numpy as np
import math
import random as rdm
import matplotlib.pyplot as plt

def inverse_demand(d, saturation=45*3):
    p = saturation - d
    return p

def table(options_pi, options_pj, mci, saturation):
    N = {}    
    for e_pj in options_pi:
        N[e_pj] = {}
        for action in options_pj:
            d = e_pj + action
            price = inverse_demand(d, saturation)
            N[e_pj][action] =  price*action - mci*action
    return N

def state_estimate(options_pi, options_pj, mci): 
    N = {}    
    for e_pj in options_pi:
        N[e_pj] = {}
        for action in options_pj:
            d = e_pj + action
            price = inverse_demand(d, saturation)
            N[e_pj][action] =  price*action - mci*action
    return N


'''         
def cost(bid, params):
    c = params[0]*bidˆˆ2 + params[1]*bid + params[2]
    return c
'''

N=4
#cost (manual)
mca = 25
mcb = 25
mcc = 25
mcd = 25
saturation = 30*N

#choose pa, pb
options_pa = [10., 12.5, 15, 17.5, 20., 21.67, 22.5, 25, 27.5, 30]
options_pb = [10., 12.5, 15, 17.5, 20., 21.67, 22.5, 25, 27.5, 30]
options_pc = [10., 12.5, 15, 17.5, 20., 21.67, 22.5, 25, 27.5, 30]
options_pd = [10., 12.5, 15, 17.5, 20., 21.67, 22.5, 25, 27.5, 30]
'''
options_pd = list(np.arange(15,61)
options_pd = list(np.arange(15,61)
options_pd = np.arange(15,61)
options_pd = np.arange(15,61)
'''

#options_pb = [6., 8., 10., 12., 14., 16., 18., 20]

#bid_pa = [rdm.choice(options_pa)]1
#bid_pb = [rdm.choice(options_pb)]

A = table(options_pa, options_pb, mca, saturation)
B = table(options_pb, options_pa, mcb, saturation)
C = table(options_pc, options_pc, mcc, saturation)
D = table(options_pd, options_pd, mcd, saturation)

BanditA = {}
BanditA = {options:0 for options in options_pa}
BanditB = {}
BanditB = {options:0 for options in options_pb}
BanditC = {}
BanditC = {options:0 for options in options_pc}
BanditD = {}
BanditD = {options:0 for options in options_pd}

total_iters=5000
gamma = 0
epsilon = 1.
alpha = 0.1

RewA = []
RewB = []
RewC = []
RewD = []

BidA = []
BidB = []
BidC = []
BidD = []

P=[]

for i in range(total_iters):
    print ('iter', i)
    
    max_bid_pa = list(BanditA.keys())[list(BanditA.values()).index(max(BanditA.values()))]
    rndm_bid_pa = rdm.choice(options_pa)
    if rdm.uniform(0,1) <= (1-epsilon):
        bid_pa = max_bid_pa
        print ("max A", bid_pa)
    else:
        bid_pa = rndm_bid_pa
        print ("random A", bid_pa)

    max_bid_pb = list(BanditB.keys())[list(BanditB.values()).index(max(BanditB.values()))]
    rndm_bid_pb = rdm.choice(options_pb)
    if rdm.uniform(0,1) <= (1-epsilon):
        bid_pb = max_bid_pb
        print ("max B", bid_pb)
    else:
        bid_pb = rndm_bid_pb
        print ("random B", bid_pb)
    
    max_bid_pc = list(BanditC.keys())[list(BanditC.values()).index(max(BanditC.values()))]
    rndm_bid_pc = rdm.choice(options_pc)
    if rdm.uniform(0,1) <= (1-epsilon):
        bid_pc = max_bid_pc
        print ("max C", bid_pc)
    else:
        bid_pc = rndm_bid_pc
        print ("random C", bid_pc)
    
    max_bid_pd = list(BanditD.keys())[list(BanditD.values()).index(max(BanditD.values()))]
    rndm_bid_pd = rdm.choice(options_pd)
    if rdm.uniform(0,1) <= (1-epsilon):
        bid_pd = max_bid_pd
        print ("max D", bid_pd)
    else:
        bid_pd = rndm_bid_pd
        print ("random D", bid_pd)
        
    d = bid_pa + bid_pb + bid_pc + bid_pd
    d = math.sqrt(bid_pa**2 + bid_pb**2 + bid_pc**2 + bid_pd**2)
    saturation = 30*N
    saturation = 25*N
    p = inverse_demand(d,saturation)    
    print ('price', p)
    ra = p*bid_pa - mca*bid_pa
    rb = p*bid_pb - mcb*bid_pb
    rc = p*bid_pc - mcb*bid_pc
    rd = p*bid_pd - mcb*bid_pd
    
    BanditA[bid_pa] = (1 - alpha)*BanditA[bid_pa] + alpha*(ra + gamma*max_bid_pa)
    BanditB[bid_pb] = (1 - alpha)*BanditB[bid_pb] + alpha*(rb + gamma*max_bid_pb)
    BanditC[bid_pc] = (1 - alpha)*BanditC[bid_pc] + alpha*(rc + gamma*max_bid_pc)
    BanditD[bid_pd] = (1 - alpha)*BanditD[bid_pd] + alpha*(rd + gamma*max_bid_pd)
    epsilon = epsilon - epsilon*(float((i*0.005))/float(total_iters))
    print ('epsilon', epsilon)
    
    RewA.append(ra)
    RewB.append(rb)
    RewC.append(rc)
    RewD.append(rd)
    
    BidA.append(bid_pa)
    BidB.append(bid_pb)
    BidC.append(bid_pc)
    BidD.append(bid_pd)
    P.append(p)

plt.figure()
plt.title('Reward trajectory for 4 n-armed bandits \nExploration: decaying, Alpha: 0.2, Discount Factor: 0.9')
plt.plot(RewA,'r--', label = 'Agent A')
plt.plot(RewB, 'b--', label = 'Agent B')
plt.plot(RewC, 'g--', label = 'Agent C')
plt.plot(RewD, 'y--', label = 'Agent D')
plt.ylabel('Net Revenue')
plt.xlabel('Iterations')
plt.legend()
plt.show()

plt.figure()
plt.title('Bid trajectory for 4 n-armed bandits \nExploration: 0.975-greedy, Alpha: 0.2, Discount Factor: 0.9')
plt.plot(BidA,'r--', label = 'Agent A')
plt.plot(BidB, 'b--', label = 'Agent B')
plt.plot(BidC, 'g--', label = 'Agent C')
plt.plot(BidD, 'y--', label = 'Agent D')
plt.ylabel('Qty Bid')
plt.xlabel('Iterations')
plt.legend()
plt.show()

plt.figure()
plt.title('Price')
plt.plot(P,'k--', label = 'Cournot Price')
plt.ylabel('Price')
plt.xlabel('Iterations')
plt.legend()
plt.show()