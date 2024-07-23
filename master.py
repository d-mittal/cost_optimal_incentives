#import random
#from numpy.random import choice
import time

import itertools
import multiprocessing as mp
from one_run_net import one_run_network
#from one_run_wm import one_run_well_mixed
import numpy as np
#from ray.util.multiprocessing.pool import Pool

from gen_gamma import gen_gamma
from gen_net import gen_net
import matplotlib.pyplot as plt
from strategy_name import strategy_name

#import sys
#import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname("one_run.py"), '..')))
#print(sys.path)



def moving_average(series, window_size):
    """
    Calculate the moving average of a given time series using a specified window size.
    """
    cumsum = [0] * (len(series) + 1)
    for i in range(1, len(series) + 1):
        cumsum[i] = cumsum[i - 1] + series[i - 1]
    moving_averages = []
    for i in range(window_size, len(series) + 1):
        moving_averages.append((cumsum[i] - cumsum[i - window_size]) / window_size)
    return moving_averages

def find_closest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))




    
#%%

if __name__ == '__main__':
    
    
    st = time.time()

    p = mp.Pool(processes=15)
    #p=Pool()
    
    N=1000
    z=20 
    
    
    V=10
    penalty=1
    
    
    #parameters of gamma distribution
    mu=0
    sigma=0.3
    
    
    #size of intervention
    phi=0.25
    
    #error in decision making
    beta=10
    
    # homophily
    p_in=0.9
    
    
    
    
    num_of_iter=10*N
    
    num_of_networks=25
    num_of_replicates=20
    num_of_gamma_configs=25
    
       
    #choose network topology [1-ER, 2-WS, 3-BA, 41 and 43-segregated networks] 
    network=3
    #network_list=[1,2,3,41,43]
    network_list=[network]
    
    
    #sequence_type=0 for random, 1 for amenable first, 2 for resistant first
    strategy_type=20
    #strategy_list=[0,10,11,20,21,30,31,40,41]
    strategy_list=[strategy_type]
    
    # gamma distribution=0 for normal, 1 for uniform, 2 for bimodal
    gamma_dis=1
    #gamma_dis_list=[0,1,2]
    gamma_dis_list=[gamma_dis]
    
    
    
    gamma_dis_labels=['Normal','Uniform','Bimodal']
    
    gamma_string='Gamma distribution: '+gamma_dis_labels[gamma_dis]
   
    
    
    network_labels = {1: "Erdos-Reyni Network", 2: "Watts-Strogatz Network", 3: "Barabasi-Albert Network", 41: "Homophilic Homogenous Network", 43: "Homophilic Heterogenous Network"}
    topology_string = network_labels.get(network, "Unknown Network")
       
    
     
    strategy_string=strategy_name(strategy_type)+' strategy'
    
    
    print('Intervention size: ' +str(int(phi*100))+ '%')

    
    ws_p=0.25
    


    gamma_range=np.linspace(mu-3*sigma, mu+3*sigma,100)
    
    
        
    input_list= np.array([(N,mu,sigma,gamma_range,gamma_dis,network) for gamma_dis,ii in itertools.product(gamma_dis_list,range(num_of_gamma_configs))], dtype=object)
    #input_list= np.array([(N,mu,sigma,gamma_range,gamma_dis,network) for i in range(num_of_gamma_configs)], dtype=object)
    
    gamma_list=p.starmap(gen_gamma,input_list)
    
       
       
    
    rnd=666
    rnd_list=range(666,666+num_of_networks)
    input_list= np.array([(network,N,z,ws_p,rnd,strategy_type,p_in) for network,rnd in itertools.product(network_list,rnd_list)], dtype=object)
    A_list,criteria_value_list=zip(*p.starmap(gen_net,input_list))
    
    input_list = np.array([(N,num_of_iter,num_of_replicates,beta,A_list[ii],V,gamma_list[i],penalty,strategy_type,phi,criteria_value_list[ii]) for i,ii in itertools.product(range(len(gamma_list)),range(len(A_list)))],dtype=object)
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Initialization time:', elapsed_time, 'seconds')
    
    xa, incentive_cost,Gini,time_taken =zip(*p.starmap(one_run_network,input_list))
        
        
        
    
    p.close()
    p.join()
    
    
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    #with open('results.npy', 'wb') as f:
        #np.save(f, results)
    print(gamma_string)
    print(topology_string) 
    print(strategy_string)
    #%%
    xa=np.array(xa)
    xa.shape=(num_of_gamma_configs*num_of_networks,num_of_iter)
    incentive_cost=np.array(incentive_cost)
    incentive_cost.shape=(num_of_gamma_configs*num_of_networks,num_of_iter)
    Gini=np.array(Gini)
    #Gini.shape=(num_of_gamma_configs*num_of_networks)
    time_taken=np.array(time_taken)
    #%%
    
    ## plotting time series of norm abandonemnt and incentives given out for a given strategy 
    
    plt.style.use("default")
    # correlation between choice and preferene
    
    
    temp=xa
      
    
    mean=np.median(temp,axis=0)
    percentile_95=np.percentile(temp,95,axis=0)
    percentile_5=np.percentile(temp,5,axis=0)
    
    
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.plot(np.array(range(num_of_iter))/N, mean,label='Median')
    ax2.fill_between(np.array(range(num_of_iter))/N,percentile_5,percentile_95, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=0,label='5-95 percentile band')
    
    ax2.axhline(y=0.95,linewidth=0.2,linestyle='--')
    ax2.axvline(x=np.nanmean(time_taken),color='k',linewidth=0.6)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Norm abandonment')
    #ax2.tick_params(bottom=True, top=True, left=True, right=True,direction="in")
    print('average Time Taken for 95% adoption = '+str(round(np.nanmean(time_taken),2)))
    print('average Gini coeff = '+str(round(np.nanmean(Gini),2)))
    
    temp=incentive_cost
    mean=np.mean(temp,axis=0)
    
    moving_mean=moving_average(mean, int(N/10))
    end_of_intervention=find_closest_index(moving_mean, 10**(-1))/N
    
    #ax.axvline(x=end_of_intervention,color='k',linewidth=0.6,linestyle='--')
    percentile_95=np.nanpercentile(temp,95,axis=0)
    percentile_5=np.nanpercentile(temp,5,axis=0)
    
    
    
    #fig, ax = plt.subplots()
    ax.plot(np.array(range(num_of_iter))/N, mean,label='Mean',c='mediumseagreen')
    ax.fill_between(np.array(range(num_of_iter))/N,percentile_5,percentile_95, alpha=0.2,edgecolor='mediumseagreen', facecolor='mediumseagreen',linewidth=0,label='5-95 percentile band')
    ax.set_xlabel('Time (in generations)')
    ax.set_ylabel('Incentive')
    ax.tick_params(bottom=True, top=True, left=True, right=True,direction="in")
    ax.set_title(topology_string+', '+gamma_string+', \n'+strategy_string+',intervention size ' +str(phi*100)+ ' %')
    print('average Total Cost = '+str(round(sum(mean),1)))
    
    #%%
    #rate of adoption plots
    
    
    temp=xa
      
    window_size=200
    mean=np.median(temp,axis=0)
    rate=np.array(moving_average(np.diff(mean), window_size))
    
    mean=np.array(moving_average(mean,window_size)[1:])
    target_rate=np.array([phi/N*np.exp(-t/N) for t in range(int(window_size/2),num_of_iter-int(window_size/2))])
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(np.array(range(int(window_size/2),num_of_iter-int(window_size/2)))/N, rate-target_rate,label='endogenous rate')
    ax.plot(np.array(range(int(window_size/2),num_of_iter-int(window_size/2)))/N, target_rate,c='r',label='exogenous rate')
    
    
    
    ax2.axhline(y=0.95,linewidth=0.2,linestyle='--')
    ax.axvline(x=np.nanmean(time_taken),color='k',linewidth=0.6)
    
    
    ax.set_ylabel('Rate of norm abandonment')
    
    
    ax2.plot(np.array(range(int(window_size/2),num_of_iter-int(window_size/2)))/N, mean,label='',c='mediumseagreen',linewidth=0.8,linestyle='--')
    
    ax.set_xlabel('Time (in generations)')
    ax2.set_ylabel('Norm abandonement (shown in green)')
    ax.tick_params(bottom=True, top=True, left=True, right=False,direction="in")
    ax.set_title(topology_string+', '+gamma_string+', \n'+strategy_string+',intervention size = ' +str(int(phi*100))+  '%')
    ax.legend(loc='center right')
    #%%
    #acceleration  of adoption plots
    
    
    # m=2
    # m1=(1+m)/2
    # acc=np.array(moving_average(np.diff(rate), window_size*m))
    # fig, ax = plt.subplots()
    # ax.plot(np.array(range(int(window_size*m1),num_of_iter-int(window_size*m1)))/N, acc,c='r',label='Median')
    # ax.tick_params(bottom=True, top=True, left=True, right=True,direction="in")
    # ax.set_xlabel('Time (in generations)')
    # ax.set_ylabel('Acceleration of norm abandonment')