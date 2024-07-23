# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:35:33 2024

@author: dmittal
"""

import numpy as np
def gaussian(x, mu, sig):
    return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))/(sig*(2*np.pi)**0.5)

def bimodal(x,frac,mu1,sig1,mu2,sig2):
    return frac*gaussian(x,mu1,sig1)+(1-frac)*gaussian(x,mu2,sig2)
def powerlaw(x, gamma):
    return np.power(x,-gamma)

def gen_gamma(N,mu,sigma,gamma_range,gamma_dis,network):
    
    if gamma_dis==0:
        gamma_pdf=[gaussian(x,mu,sigma) for x in gamma_range]
    elif gamma_dis==1:
        gamma_pdf=np.ones(len(gamma_range))/len(gamma_range)
    else:
        #gamma_pdf=[bimodal(x,0.5,mu-sigma,sigma/2,mu+sigma,sigma/2) for x in gamma_range]
        gamma_pdf=[bimodal(x,0.5,mu-1.5*sigma,sigma/2,mu+1.5*sigma,sigma/2) for x in gamma_range]
        
    gamma_pdf=gamma_pdf/sum(gamma_pdf)
    gamma=np.random.choice(gamma_range,N,p=gamma_pdf)
    
    if network>40:
        gamma.sort()
        gamma1 = np.array([gamma[i] for i in range(int(N/2))])
        np.random.shuffle(gamma1)
        gamma2=np.array([gamma[i+int(N/2)] for i in range(int(N/2))])
        np.random.shuffle(gamma2)
        gamma = np.concatenate((gamma1, gamma2))
        
    return gamma