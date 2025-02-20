# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:24:36 2025

@author: dmittal
"""

import numpy as np
from numba import njit

@njit(fastmath=True)
def comb(n,k):
    numerator = np.arange(n, n - k, -1)
    denominator = np.arange(1, k + 1)

    # Calculate the combinations
    combinations = np.prod(numerator / denominator)
    return combinations

def get_group_pdfs(crit_list, crit_pdf, target_fraction, direction=0):
    """
    Splits a discrete probability mass function (PDF) into two groups (target and non-target)
    so that the target group contains a total probability mass equal to target_fraction.
    
    For direction =0:
      - Target group: All individuals with values greater than a threshold,
        plus a fraction of those at the threshold, such that the target mass equals target_fraction.
      - Non-target group: The remaining individuals.
    
    For direction = 1:
      - Target group: All individuals with values lower than a threshold,
        plus a fraction of those at the threshold.
    
    Parameters:
      crit_list : array-like
          Sorted discrete values (assumed in ascending order).
      crit_pdf : array-like
          Corresponding probability masses (must sum to 1).
      target_fraction : float
          The desired cumulative mass for the target group (e.g., 0.2 for 20%).
      direction : str, optional
          "high" to target high values or "low" to target low values.
    
    Returns:
      target_pdf : numpy array
          Normalized probability mass function for the target group.
      non_target_pdf : numpy array
          Normalized probability mass function for the non-target group.
    """
    crit_arr = np.array(crit_list)
    pdf_arr = np.array(crit_pdf)
    
    # Order the arrays based on the targeting direction.
    if direction == 0:
        crit_ordered = crit_arr[::-1]
        pdf_ordered = pdf_arr[::-1]
    elif direction == 1:
        crit_ordered = crit_arr
        pdf_ordered = pdf_arr
    else:
        raise ValueError("direction must be either 0 or 1")
    
    # Compute the cumulative sum in the chosen order.
    cum_sum = np.cumsum(pdf_ordered)
    idx = np.searchsorted(cum_sum, target_fraction)
    cum_before = 0 if idx == 0 else cum_sum[idx - 1]
    threshold_value = crit_ordered[idx]
    fraction_needed = (target_fraction - cum_before) / pdf_ordered[idx]
    
    # Now, split the original distribution.
    if direction == 0:
        target_mask = crit_arr > threshold_value
        non_target_mask = crit_arr < threshold_value
    else:  # direction == "low"
        target_mask = crit_arr < threshold_value
        non_target_mask = crit_arr > threshold_value

    target_mass = np.zeros_like(pdf_arr)
    non_target_mass = np.zeros_like(pdf_arr)
    
    target_mass[target_mask] = pdf_arr[target_mask]
    non_target_mass[non_target_mask] = pdf_arr[non_target_mask]
    
    # For the threshold value itself, split the mass.
    idx_thresh = np.where(crit_arr == threshold_value)[0]
    if len(idx_thresh) > 0:
        i = idx_thresh[0]
        target_mass[i] = fraction_needed * pdf_arr[i]
        non_target_mass[i] = (1 - fraction_needed) * pdf_arr[i]
    
    # Normalize to form valid PDFs.
    target_pdf = target_mass / np.sum(target_mass) if np.sum(target_mass) > 0 else target_mass
    non_target_pdf = non_target_mass / np.sum(non_target_mass) if np.sum(non_target_mass) > 0 else non_target_mass
    
    return target_pdf, non_target_pdf

@njit(fastmath=True)
def prob_of_adoption(o,k,x,w):
    
      
    temp=0
    for j in range(k+1):
        marginal_payoff= o+w*(2*j-k)
                
        if marginal_payoff >= 0:
        
        #temp+=comb(int(k), int(j))*np.power(x,j)*np.power(1-x,k-j)*p1
            temp += comb(k, j) * (x**j) * ((1 - x)**(k - j))
    
    #temp=s1*p_k*p_o
    return temp



@njit(fastmath=True)
def avg_prob_adoption(k_pdf, k_list, gamma_pdf, gamma_list, x, w):
    s = 0.0
    if x>0:
        for i in range(gamma_list.shape[0]):
            if gamma_pdf[i] != 0.0:  # only proceed if gamma_pdf is non-zero
                for j in range(k_list.shape[0]):
                    if k_pdf[j] != 0.0:  # only call boop if k_pdf is non-zero
                        s += prob_of_adoption(gamma_list[i], k_list[j], x, w) * k_pdf[j] * gamma_pdf[i]
    return s