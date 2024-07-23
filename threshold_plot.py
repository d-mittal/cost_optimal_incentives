# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:24:53 2024

@author: dmittal
"""

import time

import itertools
import multiprocessing as mp
#from one_run_wm import one_run_well_mixed
import numpy as np
#from ray.util.multiprocessing.pool import Pool

from gen_gamma import gen_gamma
from gen_net import gen_net
import matplotlib.pyplot as plt
from strategy_name import strategy_name
from thresholds import thresholds
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
#%%
    

if __name__ == '__main__':
    
    
    st = time.time()

    p = mp.Pool(processes=15)
    #p=Pool()
    
    N=500
    z=10 
    
    
    V=5
    penalty=1
    
    
    #parameters of gamma distribution
    mu=0
    sigma=0.3
    
    
    #size of intervention
    phi=0.2
    
       
    # homophily
    p_in=0.9
    
       
       
    num_of_networks=50
    
    num_of_gamma_configs=50
    
       
    #choose network topology [1-ER, 2-WS, 3-BA, 41 and 43-segregated networks] 
    network=3
    #network_list=[1,2,3,41,43]
    network_list=[network]
    
    
    #sequence_type=0 for random, 1 for amenable first, 2 for resistant first
    strategy_type=0
    #strategy_list=[0,10,11,20,21,30,31,40,41]
    strategy_list=[strategy_type]
    
    # gamma distribution=0 for normal, 1 for uniform, 2 for bimodal
    gamma_dis=2
    #gamma_dis_list=[0,1,2]
    gamma_dis_list=[gamma_dis]
    
    
    
    gamma_dis_labels=['Normal','Uniform','Bimodal']
    
    gamma_string='Gamma distribution: '+gamma_dis_labels[gamma_dis]
   
    print(gamma_string)
    
    network_labels = {1: "Erdos-Reyni Network", 2: "Watts-Strogatz Network", 3: "Barabasi-Albert Network", 41: "Homophilic Homogenous Network", 43: "Homophilic Heterogenous Network"}
    topology_string = network_labels.get(network, "Unknown Network")
    print(topology_string)    
    
     
    strategy_string=strategy_name(strategy_type)+' strategy'
    print(strategy_string)
    
    print('intervention size: ' +str(int(phi*100))+ '%')

    
    ws_p=0.25
    


    gamma_range=np.linspace(mu-3*sigma, mu+3*sigma,100)
    
    
        
    input_list= np.array([(N,mu,sigma,gamma_range,gamma_dis,network) for gamma_dis,ii in itertools.product(gamma_dis_list,range(num_of_gamma_configs))], dtype=object)
    #input_list= np.array([(N,mu,sigma,gamma_range,gamma_dis,network) for i in range(num_of_gamma_configs)], dtype=object)
    
    gamma_list=p.starmap(gen_gamma,input_list)
    
       
       
    
    rnd=666
    rnd_list=range(666,666+num_of_networks)
    input_list= np.array([(network,N,z,ws_p,rnd,strategy_type,p_in) for network,rnd in itertools.product(network_list,rnd_list)], dtype=object)
    A_list,criteria_value_list=zip(*p.starmap(gen_net,input_list))
    
    input_list = np.array([(N,A_list[ii],V,gamma_list[i],penalty,strategy_type,phi,criteria_value_list[ii]) for i,ii in itertools.product(range(len(gamma_list)),range(len(A_list)))],dtype=object)
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Initialization time:', elapsed_time, 'seconds')
    
    threshold_list,target_threshold =zip(*p.starmap(thresholds,input_list))
    #freq,target_freq =zip(*p.starmap(thresholds,input_list))
        
        
        
    
    p.close()
    p.join()
    
    
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
#%%
    threshold_list=np.array(threshold_list)
    threshold_list.shape=(N*num_of_gamma_configs*num_of_networks)
    target_threshold=np.array(target_threshold)
    target_threshold.shape=(int(N*num_of_gamma_configs*num_of_networks*phi))
    

#%%

    
    fig,ax=plt.subplots()
    num_of_bins=101
    bins = np.linspace(0,1,num_of_bins)
    ax.hist(threshold_list,bins=bins, alpha=0.5)
    ax.hist(target_threshold,bins=bins, alpha=0.65, label='Target population') 
    ax.set_title(topology_string+', '+gamma_string+', \n'+strategy_string+', intervention size = ' +str(int(phi*100))+ '%')
    ax.legend()
    ax.set_xlabel('Threshold')
    ax.axvline(x=0.5,linewidth=0.5,linestyle='--',c='k')
    #ax.scatter(threshold_list)
    
#%%
    total_freq=N*num_of_gamma_configs*num_of_networks
    freq=bin_frequencies(threshold_list, bins)/total_freq
    target_freq=bin_frequencies(target_threshold,bins)/total_freq
    
    fig,ax=plt.subplots()
    #ax.plot(bins,freq)
    plt.figure(figsize=(10, 6))
    bars = ax.bar(bins, freq,width=0.01 ,color='blue',edgecolor='black', linewidth=0.5)
    alphas=np.zeros(num_of_bins)
    for i in range(num_of_bins):
        if freq[i]>0:
            alphas[i]=target_freq[i]/freq[i]
        else:
            alphas[i]=0
    alphas=alphas/np.max(alphas)
# Apply different alpha values to each bar
    for bar, alpha in zip(bars, alphas):
        bar.set_facecolor((0, 0, 1, alpha))
        bar.set_edgecolor((0,0,0,0.25))
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Frequency')
    ax.set_title(topology_string+', '+gamma_string+', \n'+strategy_string+', phi = ' +str(int(phi*100))+  '%')
    
    
    


