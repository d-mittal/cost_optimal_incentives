# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 02:53:08 2021

@author: Dhruv
"""


import numpy as np
from numba import njit



@njit(fastmath=True)
def find_closest_index(arr, target):
    """
    Find the index of the element in 'arr' that is closest to the target value.
    """
    closest_idx = 0
    min_diff = abs(arr[0] - target)

    for i in range(1, len(arr)):
        diff = abs(arr[i] - target)
        if diff < min_diff:
            min_diff = diff
            closest_idx = i

    return closest_idx

@njit(fastmath=True)

def gini(values):
    """
    Calculate the Gini coefficient of a list or array using Numba.

    Parameters:
    values (list or np.ndarray): A list or array of numerical values.

    Returns:
    float: The Gini coefficient, a number between 0 (perfect equality)
           and 1 (perfect inequality).
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return 0.0

    # Sort values
    sorted_values = np.sort(values)

    # Compute cumulative sum
    cumulative_sum = np.zeros(n)
    cumulative_sum[0] = sorted_values[0]
    for i in range(1, n):
        cumulative_sum[i] = cumulative_sum[i - 1] + sorted_values[i]

    # Calculate Gini coefficient
    total = cumulative_sum[-1]
    if total !=0:
        gini = (n + 1 - 2 * np.sum(cumulative_sum) / total) / n
    else:
        gini=np.nan
    return gini


@njit(fastmath=True)
def one_run_network(N, num_of_iter, num_of_replicates, A, gamma, penalty, strategy_type, phi, criteria_value, req):
    """
    Simulates multiple replicates of adoption dynamics for a given population configuration and  one run of a network-based dynamic process over multiple iterations and replicates.
    The simulation updates node "choices" based on a marginal payoff that depends on parameters
    such as  preference (gamma), and penalty. It also computes incentives ("bribes") and tracks the evolution
    of the adoption fraction in the network.
    
    Parameters:
    - N: Number of nodes in the network.
    - num_of_iter: Total number of iterations for the simulation.
    - num_of_replicates: Number of independent simulation replicates.
    - A: Adjacency matrix (2D numpy array) representing network connectivity.
    - V: A parameter representing value/benefit.
    - gamma: An array of node-specific values (e.g., potential benefits).
    - penalty: A penalty value affecting the marginal payoff.
    - strategy_type: An integer that determines which strategy is used to compute criteria and update dynamics.
    - phi: Fraction of nodes to target initially (determines the size of the target population).
    - criteria_value: Pre-computed criteria values for nodes (can be overridden based on strategy_type).
    - req: A threshold requirement for the final adoption fraction.
    
    Returns:
    - A tuple with average incentive cost per node, average time taken for reaching required level of adoption, average final adoption fraction,
      and two Gini coefficients (one for the target population and one for all nodes).
    """
    
    # Build a list of neighbor indices for each node and compute node degrees.
    neighbours = []
    degree = np.empty(N)
    for i in range(N):
        # Extract row i of the adjacency matrix to find connected nodes.
        values = A[:][i]
        # List comprehension: find all indices where there's an edge (value equals 1).
        temp = [i for i, x in enumerate(values) if x == 1]
        neighbours.append(temp)
        # Degree of node i is the number of its neighbors.
        degree[i] = len(temp)

    # Update criteria_value based on the strategy.
    if strategy_type == 10 or strategy_type == 11:
        # Strategy: use gamma directly as the criteria.
        criteria_value = gamma
    elif strategy_type == 20 or strategy_type == 21:
        # Strategy: use node degree as the criteria.
        criteria_value = degree
    elif strategy_type == 50:
        # Strategy: compute criteria as a ratio involving node degree, penalty, and gamma.
        criteria_value = (degree * penalty) / (- gamma + degree * penalty)
        
    
    # Determine the number of nodes to target initially.
    m = int(phi * N)

    # For non-random strategies (strategy_type > 0), sort nodes based on the criteria.
    if strategy_type > 0:
        if strategy_type % 10 == 0:
            # For strategy types ending in 0, sort in ascending order.
            sorted_indices = np.argsort(criteria_value)
        else:
            # Otherwise, sort in descending order.
            sorted_indices = np.argsort(-1 * criteria_value)

    # Select the top m nodes as the target population.
    target_pop = sorted_indices[-m:]
    
    total_cost = 0  # Initialize total cost for incentives.
    # Initialize an array to store the bribe (incentive) for each node.
    bribe = np.zeros(N)
    if strategy_type != 0:
        # For non-random strategies, compute marginal payoff and assign bribes.
        for agent in target_pop:
            k = degree[agent]
            marginal_payoff = gamma[agent] - penalty * k
            # If the marginal payoff is negative, the node requires an incentive.
            if marginal_payoff < 0:
                incentive = -marginal_payoff + 1
                total_cost += incentive
                bribe[agent] = incentive
        
        # Calculate inequality in incentives using the Gini coefficient.
        G = gini(bribe[target_pop])
        G_all = gini(bribe)

    # Initialize arrays to record outcomes across replicates.
    G_1 = np.zeros(num_of_replicates, dtype=np.float64)
    G_all_1 = np.zeros(num_of_replicates, dtype=np.float64)
    X = np.zeros(num_of_replicates, dtype=np.float64)  # Final adoption fraction in each replicate
    time_taken = np.zeros(num_of_replicates, dtype=np.float64)
    total_incentive_cost = np.zeros(num_of_replicates, dtype=np.float64)
    
    # Temporary array to track adoption changes for convergence checking.
    xtemp = np.zeros(N, dtype=np.float64)
    
    # Run the simulation for each replicate.
    for r in range(num_of_replicates):
        # x1 records the evolution of the adoption fraction over iterations.
        x1 = np.zeros(num_of_iter, dtype=np.float64)
        xa = 0  # Running total of net adoptions.
        # Initialize each node's choice: 0 (non-adopter) or 1 (adopter).
        choices = np.zeros(N, dtype=np.int8)
        
        # For random strategy (strategy_type == 0), choose a random target population.
        if strategy_type == 0:
            target_pop = np.random.choice(N, size=m, replace=False)
            total_cost = 0
            bribe = np.zeros(N)
            for agent in target_pop:
                k = degree[agent]
                marginal_payoff = gamma[agent]- penalty * k
                if marginal_payoff < 0:
                    incentive = -marginal_payoff + 1
                    total_cost += incentive
                    bribe[agent] = incentive
            # Record the inequality measures for this replicate.
            G_1[r] = gini(bribe[target_pop])
            G_all_1[r] = gini(bribe)
        
        # Set the initial choices for target nodes to 1 (adopters).
        choices[target_pop] = 1
        
        # Generate a random sequence of node indices for updating.
        sequence = np.random.choice(N, size=num_of_iter, replace=True)
        
        xa = m  # Start with m adoptions.
        cc = 0  # Counter for iterations.
        # Loop over generations, each consisting of N updates.
        for gen in range(int(num_of_iter / N)):
            for i in range(N):
                agent = sequence[cc]
                k = degree[agent]
                initial_choice = choices[agent]
                final_choice = initial_choice
                
                # Sum the adoption choices of the agent's neighbors.
                g = sum([choices[ii] for ii in neighbours[agent]])
                # Compute marginal payoff: benefit minus penalty adjusted by neighbor influence.
                marginal_payoff = gamma[agent] - penalty * (k - 2 * g)
                # Agent adopts (final_choice = 1) if marginal payoff is non-negative.
                final_choice = int(marginal_payoff >= 0)
                choices[agent] = final_choice
                # Update net adoption change.
                xa += final_choice - initial_choice
                
                # Record the current adoption fraction.
                x1[cc] = xa / N
                xtemp[i] = xa / N
                cc += 1
            
            # Check for convergence: if variation in adoption fraction is very low, break early.
            if (np.std(xtemp) < 0.001) or gen == int(num_of_iter / N) - 1:
                break
        
        # Record the final adoption fraction for this replicate.
        X[r]=np.mean(xtemp)
        if X[r] >= req:
            
            # Find the index in the moving average that is closest to the requirement.
            time_to_req=find_closest_index(x1[0:cc], req)/N
            time_taken[r] = time_to_req
            total_incentive_cost[r] = total_cost / N
        else:
            # If requirement is not met, mark the outcome as undefined (NaN).
            time_taken[r] = np.nan
            total_incentive_cost[r] = np.nan

    # Return results based on strategy type.
    if strategy_type == 0:
        # For random strategy: return average cost per node, average time to req, final adoption,
        # and average Gini coefficients for the target and overall.
        return (np.nanmean(total_incentive_cost), np.nanmean(time_taken), 
                np.percentile(X,0.25), np.nanmean(G_1), np.nanmean(G_all_1))
    else:
        # For other strategies: return total cost per node, average time, final adoption fraction,
        # and Gini coefficients computed earlier.
        return (total_cost / N, np.nanmean(time_taken), np.percentile(X,0.25), G, G_all)