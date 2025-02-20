# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:30:14 2025

@author: dmittal
"""

import numpy as np
import networkx as nx
from segregated_networks import stochastic_block_model 
from segregated_networks import preferential_attachment_homophily
from tune_net import tunable_graph

def gen_net(network, N, z, ws_p, rndseed, strategy_type, p_in):
    """
    Generates a network graph using various models and computes node criteria values 
    based on a chosen strategy.
    
    Parameters:
    - network: Integer code specifying which network model to generate:
        1  -> Erdos-Renyi Graph,
        2  -> Watts-Strogatz Graph,
        3  -> Barabasi-Albert Graph,
        41 -> Stochastic Block Model,
        43 -> Preferential Attachment Homophily,
        5  -> Powerlaw Cluster Graph,
        6  -> Graph Tunable heterogeneity .
    - N: Number of nodes in the network.
    - z: Parameter influencing average degree (or number of neighbors) or model-specific parameter.
    - ws_p: Rewiring probability (used in Watts-Strogatz and powerlaw cluster models).
    - rndseed: Seed for the random number generator for reproducibility.
    - strategy_type: Determines which network metric to compute:
        * 40 or 41: Compute clustering coefficients.
        * 30 or 31: Compute closeness centrality.
        * Otherwise: Use an array of zeros.
    - p_in: Model-specific parameter (e.g., for the stochastic block model).
    
    Returns:
    - A: The adjacency matrix of the generated graph (dense matrix of type int8).
    - criteria_value: A numpy array containing the computed metric values (or zeros if not applicable).
    """
    # Generate the network graph based on the chosen model
    if network == 1:
        # Erdos-Renyi: each edge exists with probability z/N
        c = nx.erdos_renyi_graph(N, z/N, seed=rndseed)
    elif network == 2:
        # Watts-Strogatz: small-world network with rewiring probability ws_p
        c = nx.watts_strogatz_graph(N, z, ws_p, seed=rndseed)
    elif network == 3:
        # Barabasi-Albert: preferential attachment graph
        c = nx.barabasi_albert_graph(int(N), z, seed=rndseed)
    elif network == 41:
        # Stochastic Block Model 
        c = stochastic_block_model(N, z, p_in)
    elif network == 43:
        # Preferential Attachment Homophily model
        c = nx.from_numpy_array(preferential_attachment_homophily(N, z, p_in))
    elif network == 5:
        # Powerlaw Cluster Graph: incorporates tunable clustering with a power-law degree distribution
        c = nx.powerlaw_cluster_graph(N, z, ws_p, seed=rndseed)
    elif network == 6:
        # Tunable graph model with a fixed extra parameter (here, 3)
        c = tunable_graph(N, z, ws_p, 3)
    
    # Compute node metric based on strategy_type
    if strategy_type == 40 or strategy_type == 41:
        # Compute clustering coefficient for each node
        temp = nx.clustering(c)
    elif strategy_type == 30 or strategy_type == 31:
        # Compute closeness centrality for each node
        temp = nx.closeness_centrality(c)
    
    # Assign criteria values:
    # For strategy types that don't require a metric (strategy_type < 30 or >= 50), return zeros.
    if strategy_type < 30 or strategy_type >= 50:
        criteria_value = np.zeros(N)
    else:
        # Convert the metric dictionary to a numpy array (order corresponds to node labels)
        criteria_value = np.array(list(temp.values()))
    
    # Convert the graph to its adjacency matrix
    A = nx.adjacency_matrix(c)
    A = A.todense()       # Convert from sparse matrix to dense matrix
    A = A.astype('int8')  # Cast the matrix to int8 type
    return A, criteria_value

def generate_criteria_value(A, strategy_type):
    """
    Computes node criteria values (such as clustering coefficients or closeness centrality)
    from a given adjacency matrix.
    
    Parameters:
    - A: Adjacency matrix of the graph.
    - strategy_type: Integer that specifies which metric to compute:
        * 40 or 41: Compute clustering coefficients.
        * 30 or 31: Compute closeness centrality.
        * Otherwise: Return an array of zeros.
    
    Returns:
    - criteria_value: A numpy array of the computed metric values for each node.
    """
    # Determine the number of nodes from the adjacency matrix dimensions
    N = len(A[0])
    # Reconstruct the graph from the adjacency matrix
    c = nx.from_numpy_array(A)
    
    if strategy_type == 40 or strategy_type == 41:
        # Compute clustering coefficients for the graph
        temp = nx.clustering(c)
    elif strategy_type == 30 or strategy_type == 31:
        # Compute closeness centrality for the graph
        temp = nx.closeness_centrality(c)
    
    # If the strategy type does not match any supported metric, return an array of zeros.
    if strategy_type < 30 or strategy_type >= 50:
        criteria_value = np.zeros(N)
    else:
        # Convert the metric values from the dictionary to a numpy array
        criteria_value = np.array(list(temp.values()))
    
    return criteria_value

def degree_heterogeneity(network, N, z, ws_p, rndseed, p_in):
    """
    Computes the degree heterogeneity of a generated network, defined as the standard deviation
    of the degrees of the nodes.
    
    Parameters:
    - network: Integer code specifying which network model to generate (same as in gen_net).
    - N: Number of nodes in the network.
    - z: Parameter influencing the average degree or model-specific parameter.
    - ws_p: Rewiring probability for models like Watts-Strogatz and powerlaw cluster.
    - rndseed: Seed for the random number generator for reproducibility.
    - p_in: Model-specific parameter (used in some network models).
    
    Returns:
    - A single value representing the standard deviation of node degrees (degree heterogeneity).
    """
    # Generate the network graph based on the chosen model
    if network == 1:
        c = nx.erdos_renyi_graph(N, z/N, seed=rndseed)
    elif network == 2:
        c = nx.watts_strogatz_graph(N, z, ws_p, seed=rndseed)
    elif network == 3:
        c = nx.barabasi_albert_graph(int(N), z, seed=rndseed)
    elif network == 41:
        c = stochastic_block_model(N, z, p_in)
    elif network == 43:
        c = nx.from_numpy_array(preferential_attachment_homophily(N, z, p_in))
    elif network == 5:
        c = nx.powerlaw_cluster_graph(N, z, ws_p, seed=rndseed)
    elif network == 6:
        c = tunable_graph(N, z, ws_p, 3)
        
    # Extract the degree for each node from the graph
    degrees = [d for _, d in c.degree()]
    
    # Compute and return the standard deviation of the node degrees
    return np.std(degrees, ddof=0)
