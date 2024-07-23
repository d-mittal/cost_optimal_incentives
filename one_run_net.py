# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 02:53:08 2021

@author: Dhruv
"""


#from numpy.random import choice
import numpy as np
from numba import njit


@njit(fastmath=True)
def find_closest_index(arr, target):
    closest_idx = 0
    min_diff = abs(arr[0] - target)
    
    for i in range(1, len(arr)):
        diff = abs(arr[i] - target)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    
    return closest_idx

@njit(fastmath=True)
def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference'=
    avg=np.mean(x)
    if avg==0:
        return np.nan
    else:
        mad =np.mean(np.array([abs(i-avg) for i in x]))
        # Relative mean absolute difference
        rmad = mad/avg
        # Gini coefficient
        g = 0.5 * rmad
        return g


@njit(fastmath=True)
def one_run_network(N,num_of_iter,num_of_replicates,beta,A,V,gamma,penalty,strategy_type,phi,criteria_value):
    
    
    
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
            target_pop = sorted_indices[-m:]
            #threshold=np.percentile(criteria_value,(1-phi)*100)
            #target_pop=np.array([i for i in range(N) if criteria_value[i]>=threshold])
        else:
            target_pop=sorted_indices[:m]
            #threshold=np.percentile(criteria_value,phi*100)
            #target_pop=np.array([i for i in range(N) if criteria_value[i]<=threshold])
            
    
    
    
       
    G=np.zeros(num_of_replicates ,dtype=np.float64)
    time_taken=np.zeros(num_of_replicates, dtype=np.float64)
    
    X1 = np.zeros(num_of_iter, dtype=np.float64)
    x1 = np.zeros(num_of_iter, dtype=np.float64)
    incentives=np.zeros(num_of_iter, dtype=np.float64)
    for r in range(num_of_replicates):
        xa=0
        

        if strategy_type==0:        
            target_pop=np.random.choice(N, size=int(N*phi), replace=False)
            
        choices = np.zeros(N, dtype=np.int64)
               
                                       
        sequence = np.random.choice(N, size=num_of_iter, replace=True)
        
            
        
        total_incentive_cost=0
        
        bribe=np.zeros(N)
        
        for i in range(num_of_iter):
            
            #random
            agent=sequence[i]
            k=degree[agent]
            
            
            
            initial_choice = choices[agent]
            final_choice = initial_choice
            
            
            g = sum([choices[ii] for ii in neighbours[agent]])
            
            marginal_payoff= (V*gamma[agent])- penalty*(k-2*g)
            
            if marginal_payoff<0  and agent in target_pop:
                incentive=-marginal_payoff+1
                incentives[i]+=incentive/num_of_replicates
                marginal_payoff=1
                total_incentive_cost+=incentive
                bribe[agent] +=incentive
            
            p1 = (1 + np.exp(-marginal_payoff * beta))**(-1)
            if p1 >= np.random.random():
                final_choice = 1
            else: 
                final_choice = 0
                
            
    
            choices[agent] = final_choice
            xa += final_choice - initial_choice
            #xa += final_choice - initial_choice
            x1[i] = xa /(N)
            
            
        X1+=x1/num_of_replicates
        if x1[-1]>=0.95:
            time_to_95=find_closest_index(x1, 0.95)/N
            time_taken[r]=time_to_95
        else:
            time_taken[r]=num_of_iter/N
       
        
        G[r]=gini(bribe)
        
        
        
    
    return X1,incentives,G,time_taken





