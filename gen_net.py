# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:37:53 2024

@author: dmittal
"""


import numpy as np
import networkx as nx
from segregated_networks import stochastic_block_model 
from segregated_networks import preferential_attachment_homophily


def gen_net(network,N, z, ws_p,rndseed,strategy_type,p_in):
    if network == 1:
        c = nx.erdos_renyi_graph(N, z/N,seed=rndseed)
    elif network == 2:
          c=nx.watts_strogatz_graph(N, z, ws_p, seed=rndseed)
    elif network == 3:
        c=nx.barabasi_albert_graph(int(N), z,seed=rndseed)
    elif network == 41:
        c=stochastic_block_model(N, z, p_in)
    elif network == 43:
        c=nx.from_numpy_array(preferential_attachment_homophily(N, z, p_in))
        
    if strategy_type==40 or strategy_type==41:
        temp=nx.clustering(c)
    elif strategy_type==30 or strategy_type==31:
        temp=nx.betweenness_centrality(c)
    
    if strategy_type<30:
        criteria_value=np.zeros(N)
    else:
        criteria_value=np.array(list(temp.values()))
    
    A= nx.adjacency_matrix(c)
    A=A.todense()
    A=A.astype('int8')
    return A,criteria_value
