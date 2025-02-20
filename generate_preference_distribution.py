# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:25:20 2025

@author: dmittal
"""

from scipy.stats import beta
import numpy as np



# Generates an array of gamma values (or weights) based on various probability distributions.
#
# Parameters:
#   N         - The number of gamma values to generate.
#   mu        - The central value (mean) around which values are distributed.
#   gamma_dis - An integer flag to choose the type of distribution:
#                0: Gaussian,
#                1: Uniform,
#                2: Bimodal,
#                else: Beta distribution.
#   network   - A parameter used to conditionally modify the generated gamma values if network > 40.
#   alpha     - Parameter for the Beta distribution (used symmetrically as both alpha parameters).
def gen_gamma(N, mu, network, alpha,V):
    
   
    # Beta distribution: redefine gamma_range to be within a narrow interval around mu (±0.5)
    # and shift it to [0, 1] for the beta PDF calculation.
    gamma_range = np.linspace(mu - 0.499, mu + 0.499, 100)
    gamma_pdf = beta.pdf(gamma_range - (mu - 0.5), alpha, alpha)
    
    # Normalize the PDF so that the probabilities sum to 1.
    gamma_pdf = gamma_pdf / sum(gamma_pdf)
    
    # Randomly select N values from gamma_range using the computed probabilities.
    gamma = np.random.choice(gamma_range, N, p=gamma_pdf)
    
    gamma=gamma*V
    # If the network is homophilous, then the values are reordered into two halves of resitant and amenable:
    if network > 40:
        # First, sort the gamma values.
        gamma.sort()
        # Split the sorted gamma array into two halves.
        gamma1 = np.array([gamma[i] for i in range(int(N / 2))])
        gamma2 = np.array([gamma[i + int(N / 2)] for i in range(int(N / 2))])
        # Randomly shuffle each half separately.
        np.random.shuffle(gamma1)
        np.random.shuffle(gamma2)
        # Concatenate the two halves back together.
        gamma = np.concatenate((gamma1, gamma2))
        
    # Return the generated gamma values.
    return gamma