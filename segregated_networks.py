# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:39:47 2025

@author: dmittal
"""

import networkx as nx
import numpy as np
import random

def random_subset_with_weights(weights, m):
    """
    Selects a random subset of indices of size m, where the probability of each index 
    being selected is influenced by its weight. Uses an exponential sampling trick.
    
    Parameters:
    - weights: A list or array of weights (higher weight increases the chance of selection).
    - m: The number of indices to select.
    
    Returns:
    - A list of m indices chosen based on the provided weights.
    """
    # Ensure that weights is a numpy array for vectorized operations
    weights = np.array(weights)
    
    # Initialize an array to store exponential variates
    expovariates = np.zeros(len(weights))
    
    # Generate an exponential random variable for each weight.
    # The idea is that drawing an exponential random variable with rate 1/weight 
    # biases the selection towards lower exponential samples for larger weights.
    for i in range(len(weights)):
        if weights[i] == 0:
            # If the weight is zero, assign an infinite variate so it is not selected.
            expovariates[i] = np.inf
        else:
            # Draw from an exponential distribution with scale = 1/weight.
            expovariates[i] = np.random.exponential(1.0 / weights[i])
    
    # Obtain indices that would sort the exponential variates in ascending order.
    sorted_indices = np.argsort(expovariates)
    
    # Select the first m indices, which correspond to the smallest exponential samples.
    subset_indices = sorted_indices[:m]
    
    return subset_indices.tolist()

def preferential_attachment_homophily(n, m, p_in):
    """
    Generates an adjacency matrix for a network with preferential attachment 
    and homophily properties. Nodes are divided into two groups, and the probability 
    of connecting to a node is weighted based on its degree and group similarity.
    
    Parameters:
    - n: Total number of nodes.
    - m: Number of edges to attach for each new node.
    - p_in: A parameter (between 0 and 1) indicating the tendency to connect within the same group.
    
    Returns:
    - A: An n x n numpy array representing the adjacency matrix of the generated graph.
    """
        
    
    # Initialize with a complete graph on m vertices selected from both groups:
    # Randomly choose m/2 nodes from each group (first group: 0 to n/2 - 1, second group: n/2 to n-1)
    seed1 = np.random.choice(int(n/2), size=int(m/2), replace=False)
    seed2 = np.random.choice(int(n/2), size=int(m/2), replace=False) + int(n/2)
    # Combine the seeds from both groups
    seed = np.concatenate((seed1, seed2))
    
    # Initialize an array for node degrees; set degrees of seed nodes to m-1 
    # (because they form a complete graph among themselves)
    degrees = np.zeros(n)
    degrees[seed] = m - 1 
    
    # Create an empty adjacency matrix for n nodes
    A = np.zeros([n, n])
    # For each seed node, connect it to every other seed node (forming a complete subgraph)
    for i in seed:
        # Use set difference to avoid self-loops
        A[i][list(set(seed) - {i})] = 1
        
    # Determine the list of nodes that are not in the seed set
    remaining_nodes = list(set(range(n)) - set(seed))
    # Shuffle the remaining nodes to introduce randomness in their order of attachment
    random.shuffle(remaining_nodes)
        
    group_size = int(n/2)
    # Process each remaining node to attach it to m neighbors
    for i in remaining_nodes:
        # Depending on which group the node belongs to, assign different weights for connections:
        if i < n/2:
            # For nodes in the first group, assign a higher weight (p_in) to nodes in the first half 
            # and a lower weight (1-p_in) to nodes in the second half.
            weights = np.array([p_in] * group_size + [1 - p_in] * group_size) * degrees
        else:
            # For nodes in the second group, reverse the weighting.
            weights = np.array([1 - p_in] * group_size + [p_in] * group_size) * degrees
        
        # Use the weighted random subset function to select m neighbors based on the computed weights.
        n_neighbours = random_subset_with_weights(weights, m)
        
        # Create a directed edge from node i to each of the selected neighbors.
        A[i][n_neighbours] = 1
        
        # Set the degree of node i to m (as it has m outgoing links now)
        degrees[i] = m
        
        # For each selected neighbor, update their degree and ensure the edge is bidirectional.
        for j in n_neighbours:
            degrees[j] += 1
            A[j][i] = 1

    return A

def stochastic_block_model(N, z, p_in):
    """
    Generates a Stochastic Block Model (SBM) graph using networkx.
    
    In this SBM, nodes are divided equally into two blocks. The probability of an edge 
    between nodes is determined by whether they belong to the same block or not.
    
    Parameters:
    - N: Total number of nodes in the graph.
    - z: A parameter used to scale the edge probability.
         Here, it is normalized by the block size (N/2).
    - p_in: The probability multiplier for edges within the same block.
            For edges between blocks, the probability multiplier is (1 - p_in).
    
    Returns:
    - G: A networkx Graph object representing the generated stochastic block model.
    """
    # Create block assignments: first half of nodes in block 0, second half in block 1.
    block_assignment = np.array([1 - int(i < N / 2) for i in range(N)])
    
    # Initialize an empty undirected graph.
    G = nx.Graph()
    
    # Set the probability multiplier for inter-block edges.
    p_out = 1 - p_in
    
    # Add all nodes to the graph with an attribute indicating their block assignment.
    for i in range(N):
        G.add_node(i, block=block_assignment[i])

    # Iterate over all unique pairs of nodes to potentially add an edge between them.
    for i in range(N):
        for j in range(i + 1, N):
            # Base probability factor scaled by z and normalized by the block size.
            p = z / (N / 2)
            
            if block_assignment[i] == block_assignment[j]:
                # If nodes i and j belong to the same block, edge probability is scaled by p_in.
                if np.random.rand() < p * p_in:
                    G.add_edge(i, j)
            else:
                # If nodes i and j belong to different blocks, use p_out (i.e., 1 - p_in) as the multiplier.
                if np.random.rand() < p * p_out:
                    G.add_edge(i, j)

    return G
