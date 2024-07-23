# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:28:53 2024

@author: dmittal
"""

import networkx as nx
import numpy as np
import random



def random_subset_with_weights(weights, m):
    # Ensure weights is a numpy array
    weights = np.array(weights)
    
    # Replace zero weights with a very large value to simulate infinite expovariate
    #large_value = 1e10
    #safe_weights = np.where(weights == 0, large_value, weights)
    
    # Generate random values using the exponential distribution with the given weights
    expovariates=np.zeros(len(weights))
    for i in range(len(weights)):
        if weights[i]==0:
            expovariates[i]=np.inf
        else:
            expovariates[i] = np.random.exponential(1.0 / weights[i])
    
    # Sort the indices of the expovariate values
    sorted_indices = np.argsort(expovariates)
    
    # Select the first m indices
    subset_indices = sorted_indices[:m]
    
    return subset_indices.tolist()



def preferential_attachment_homophily(n, m,p_in):
    
    links_in=int(m*p_in)
    links_out=m-links_in
    
    # initialise with a complete graph on m vertices
    seed1=np.random.choice(int(n/2),size=int(m/2),replace=False)
    seed2=np.random.choice(int(n/2),size=int(m/2),replace=False)+int(n/2)
    seed=np.concatenate((seed1,seed2))
    #neighbours = [ list(set(seed) - {i}) for i in seed ]
    degrees=np.zeros(n)
    degrees[seed] =  m-1 
    A=np.zeros([n,n])
    for i in seed:
        A[i][list(set(seed)-{i})]=1
        
    remaining_nodes=list(set(range(n)) - set(seed))
    random.shuffle(remaining_nodes)
        
    for i in remaining_nodes:
        if i<n/2:
            
        
            targets_0 = random_subset_with_weights(degrees[0:int(n/2)], links_in)
            targets_1 = random_subset_with_weights(degrees[int(n/2):], links_out)
        else:
            targets_0 = random_subset_with_weights(degrees[0:int(n/2)], links_out)
            targets_1 = random_subset_with_weights(degrees[int(n/2):], links_in)
            
        n_neighbours = targets_0+[ii+int(n/2) for ii in targets_1]
        A[i][n_neighbours]=1

        degrees[i]=m
        
        # add forward-edges
        for j in n_neighbours:
            
            degrees[j] += 1
            A[j][i]=1

        

    return A


def stochastic_block_model(N,z, p_in):
    """
    Generate a stochastic block model (SBM) graph.

    Parameters:
    - N: Total number of nodes in the graph.
    - num_blocks: Number of blocks (communities).
    - p_in: Probability of an edge within the same block.
    - p_out: Probability of an edge between different blocks.

    Returns:
    - A networkx Graph object representing the generated SBM.
    """
    # Generate block assignments for nodes
    
    
    block_assignment = np.array([1-int(i<N/2) for i in range(N)])
    # Initialize graph
    G = nx.Graph()
    p_out=1-p_in
    # Add nodes with their block attributes
    for i in range(N):
        G.add_node(i, block=block_assignment[i])

    # Iterate over all pairs of nodes
    
    
    for i in range(N):
        for j in range(i + 1, N):
            
            p=z/(N/2)
            
            if block_assignment[i] == block_assignment[j]:
                
                # Nodes i and j are in the same block
                if np.random.rand() < p*p_in:
                    G.add_edge(i, j)
            else:
                # Nodes i and j are in different blocks
                if np.random.rand() < p*p_out:
                    G.add_edge(i, j)

    return G

