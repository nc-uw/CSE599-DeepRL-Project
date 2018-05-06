#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 00:07:29 2018

@author: scsingh
"""
import numpy as np
import math
import random as rdm
import matplotlib.pyplot as plt

def inverse_demand(d, saturation=90):
    p = saturation - d
    return p

def table(options_pi, options_pj, mci):
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

#cost (manual)
mca = 25
mcb = 25
#mcb = 35

#choose pa, pb
options_pa = [10., 12.5, 15, 17.5, 20., 22.5, 25, 27.5, 30]
options_pb = [10., 12.5, 15, 17.5, 20., 22.5, 25, 27.5, 30]
#options_pb = [6., 8., 10., 12., 14., 16., 18., 20]

#bid_pa = [rdm.choice(options_pa)]
#bid_pb = [rdm.choice(options_pb)]

A = table(options_pa, options_pb, mca)
B = table(options_pb, options_pa, mcb)

BanditA = {}
BanditA = {options:0 for options in options_pa}
BanditB = {}
BanditB = {options:0 for options in options_pb}

total_iters=1000
gamma = 1.
epsilon = 1.
alpha = 0.3
saturation = 90

RewA = []
RewB = []
BidA = []
BidB = []
P=[]

for i in range(total_iters):
    print ('iter', i)
    max_bid_pa = list(BanditA.keys())[list(BanditA.values()).index(max(BanditA.values()))]
    rndm_bid_pa = rdm.choice(options_pa)
    if rdm.uniform(0,1) <= (1 - epsilon):
        bid_pa = max_bid_pa
        print ("max A", bid_pa)
    else:
        bid_pa = rndm_bid_pa
        print ("random A", bid_pa)

    max_bid_pb = list(BanditB.keys())[list(BanditB.values()).index(max(BanditB.values()))]
    rndm_bid_pb = rdm.choice(options_pb)
    if rdm.uniform(0,1) <= (1 - epsilon):
        bid_pb = max_bid_pb
        print ("max B", bid_pb)
    else:
        bid_pb = rndm_bid_pb
        print ("random B", bid_pb)
    
    d = bid_pa + bid_pb
    p = inverse_demand(d,saturation)    
    print ('price', p)
    ra = p*bid_pa - mca*bid_pa
    rb = p*bid_pb - mcb*bid_pb
    
    BanditA[bid_pa] = (1 - alpha)*BanditA[bid_pa] + alpha*(ra + gamma*bid_pa)
    BanditB[bid_pb] = (1 - alpha)*BanditB[bid_pb] + alpha*(rb + gamma*bid_pb)
    epsilon = epsilon - epsilon*(float((i*0.1))/float(total_iters))
    print ('epsilon', epsilon)
    RewA.append(ra)
    RewB.append(rb)
    BidA.append(bid_pa)
    BidB.append(bid_pb)
    P.append(p)

plt.figure()
plt.plot(RewA,'r--', RewB, 'b--')
#, RewB, 'bs')
plt.show()

plt.figure()
plt.plot(BidA,'r--', BidB, 'b--')
plt.show()
        
    
    
    
            
        

