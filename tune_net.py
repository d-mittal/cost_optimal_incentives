# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:56:27 2024

@author: dmittal
"""

import numpy as np
from numba import njit
import networkx as nx


@njit(fastmath=True)
def gaussian(x, mu, sig):
    return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))/(sig*(2*np.pi)**0.5)

def combined_pdf(x, min_degree, alpha, power_law_exponent):
    """
    Combined PDF of a Poisson and a power-law distribution with a minimum degree.

    Parameters:
    - x (int): Degree value.
    - mean_degree (float): Mean degree for the Poisson component.
    - alpha (float): Interpolation factor (0 = fully Poisson, 1 = fully power-law).
    - power_law_exponent (float): Exponent for the power-law distribution.
    - min_degree (int): Minimum degree for the power-law component.

    Returns:
    - float: Probability density for degree `x`.
    """
    
    mean_degree=min_degree*2
    # Poisson probability mass function
    poisson_prob = gaussian(x,mean_degree,mean_degree/4)
    
    # Power-law (Zipf) probability mass function with shift for minimum degree
    if x >= min_degree:
        power_law_prob = (x ** -power_law_exponent) / sum([(k ** -power_law_exponent) for k in range(min_degree, 200)])
    else:
        power_law_prob = 0
    
    # Weighted combination of both distributions
    combined_prob = (1 - alpha) * poisson_prob + alpha * power_law_prob
    return combined_prob

def tunable_graph(N, min_degree=10, alpha=0.5, power_law_exponent=3):
    """
    Sample degrees from the combined Poisson and power-law distribution using rejection sampling.

    Parameters:
    - n_samples (int): Number of samples to generate.
    - mean_degree (float): Average degree for the Poisson component.
    - alpha (float): Interpolation factor (0 = fully Poisson, 1 = fully power-law).
    - power_law_exponent (float): Exponent for the power-law distribution.
    - min_degree (int): Minimum degree for the power-law component.

    Returns:
    - network generated using configuration model.
    """
    degree_range=np.linspace(1,200,200)
    degree_pdf=[combined_pdf(x,min_degree,alpha,power_law_exponent) for x in degree_range]
    degree_pdf=degree_pdf/sum(degree_pdf)
    
    degree_range=np.array([i for i in range(1,201)])
    degrees=np.random.choice(degree_range,N,p=degree_pdf)
    if sum(degrees) % 2 != 0:
        degrees[np.random.randint(0, N)] += 1
    G = nx.configuration_model(degrees, create_using=nx.Graph)
    
    
    return G






