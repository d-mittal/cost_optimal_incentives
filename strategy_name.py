# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:39:14 2024

@author: dmittal
"""
def strategy_name(strategy_type):
    if strategy_type>=20:
        if strategy_type % 10 == 0:
            strategy_string='High '
        else:
            strategy_string='Low '
            
        temp=strategy_type//10
        
        
        if temp==2:
            strategy_string+="node \ndegree"
        elif temp==3:
            strategy_string+="closeness centrality"
        elif temp==4:
            strategy_string+="local \nclustering"
        elif temp==5:
            strategy_string+='infl./cost'
        elif temp==6:
            strategy_string+='complex centrality'
    elif strategy_type==10:
        strategy_string='Amenable'
    elif strategy_type==11:
        strategy_string='Resistant'
    elif strategy_type==0:
        strategy_string='Random'

    
    return strategy_string 