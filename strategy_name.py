# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:39:14 2024

@author: dmittal
"""
def strategy_name(strategy_type):
    if strategy_type>0:
        if strategy_type % 10 == 0:
            strategy_string='High '
        else:
            strategy_string='Low '
            
        temp=strategy_type//10
        
        if temp==1:
            strategy_string+="gamma"
        elif temp==2:
            strategy_string+="node degree"
        elif temp==3:
            strategy_string+="between-ness centrality"
        elif temp==4:
            strategy_string+="local clustering"
    else:
        strategy_string='Random'
    
    return strategy_string 