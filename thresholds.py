# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:22:34 2024

@author: dmittal
"""
import numpy as np
from numba import njit


@njit(fastmath=True)
def bin_frequencies(values, bins):
    # Using numpy to digitize the values into the specified bins
    bin_indices = np.digitize(values, bins, right=True)
    
    # Using numpy's bincount to count the frequency of each bin index
    bin_counts = np.bincount(bin_indices, minlength=len(bins) + 1)
    
    # The first bin (index 0) is not used, so we can exclude it
    frequencies = bin_counts[1:]
    
    return frequencies

@njit(fastmath=True)
def thresholds(N,A,V,gamma,penalty,strategy_type,phi,criteria_value):
    
    
    
    neighbours=[] 
    degree=np.empty(N)
    for i in range(N):
        values=A[:][i]
        #neighbours = np.where(values == 1)[0]
        temp=[i for i, x in enumerate(values) if x == 1]
        neighbours.append(temp)
        degree[i]=len(temp)
    
    #rank_list=np.argsort(degree)
    
    if strategy_type==10 or strategy_type==11:
        criteria_value=gamma    
    
    elif strategy_type==20 or strategy_type==21:
        criteria_value=degree    
        
    sorted_indices = np.argsort(criteria_value)  
    m=int(phi*N)
    if strategy_type>0 :
        if strategy_type % 10 == 0:
                        
            # Select based on highest criteria measure
            target_pop = sorted_indices[-m:]
            
        else:
            # Select based on lowest criteria measure
            target_pop=sorted_indices[:m]
            
    else:
        target_pop=np.random.choice(N, size=int(N*phi), replace=False)
    
    threshold_list=np.zeros(N)
    for i in range(N):
        threshold_list[i]=0.5*(1-V*gamma[i]/(penalty*degree[i]))

    bins = np.linspace(0,0.5,51)
    target_threshold=threshold_list[target_pop]
    # freq=bin_frequencies(threshold_list, bins)
    # target_freq=bin_frequencies(target_threshold,bins)
            
    return threshold_list,target_threshold
    #return freq,target_freq