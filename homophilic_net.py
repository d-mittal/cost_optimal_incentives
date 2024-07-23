# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:48:29 2024

@author: dmittal
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from segregated_networks import stochastic_block_model 
from segregated_networks import preferential_attachment_homophily
from gen_gamma import gen_gamma
from gen_net import gen_net




N=1000
z=20
mu=1
sigma=0.25
gamma_list=[]
ws_p=0.01
p_in=0.5
gamma_dis=2
network=43
rndseed=666
strategy_type=41

gamma_range=np.linspace(mu-3*sigma, mu+3*sigma,200)
gamma=gen_gamma(N, mu, sigma, gamma_range, gamma_dis, network)

#gamma=rearrange_array(gamma)
# fig, ax = plt.subplots()
# bins = np.histogram_bin_edges(gamma, bins='auto')
# ax.hist(gamma,bins=bins, alpha=0.5, label='Histogram 1') 


A,criteria_value=gen_net(network, N, z, ws_p, rndseed, strategy_type, p_in)

G=nx.from_numpy_array(A)



#nodes_to_remove = [node for node in G.nodes if not any(node in edge for edge in G.edges)]
#print(len(nodes_to_remove))
#G_filtered = G.copy()
#G_filtered.remove_nodes_from(nodes_to_remove)
degrees = [degree for node, degree in G.degree()]

fig, ax = plt.subplots()
bins = np.histogram_bin_edges(degrees, bins='auto')
ax.hist(degrees,bins=bins, alpha=0.5, label='Histogram 1')
#ax.set_xlim(0,100)
#%%

#gamma = [elem for i, elem in enumerate(gamma) if i not in nodes_to_remove]
 
value_dict = {i: gamma[i] for i in range(len(gamma))}



# Adding the node attributes
nx.set_node_attributes(G, value_dict, 'gamma')

first_half = [0] * (N // 2)
second_half = [1] * (N // 2)
    
community_list = first_half + second_half
value_dict = {i: community_list[i] for i in range(len(gamma))}
nx.set_node_attributes(G, value_dict, 'community')
# Step 4: Draw the graph
pos = nx.circular_layout(G)  # positions for all nodes

# Extract the node values to use for coloring
#node_values = [data['gamma'] for _, data in G_filtered.nodes(data=True)]
node_values=gamma
# Normalize node values for colormap
norm = plt.Normalize(min(node_values), max(node_values))

# Get colors from the 'bwr' colormap
node_colors = cm.bwr(norm(node_values))


# Draw the graph
plt.figure(figsize=(8, 8))
nx.draw(
    G,pos=pos,
    node_color=node_colors, 
    with_labels=False, 
    node_size=75, 
    cmap=cm.bwr, 
    edge_color='gray',
    alpha=0.3
)

# Display the color bar
sm = plt.cm.ScalarMappable(cmap=cm.bwr, norm=norm)
sm.set_array([])
#plt.colorbar(sm, label='preference for change( gamma)')

#plt.title("Watts-Strogatz Graph with Node Colors Corresponding to Values")
plt.show()
#%%

print("average degree of group 1= "+str(np.mean(degrees[0:int(N/2)])))
print("average degree of group 2= "+str(np.mean(degrees[int(N/2):])))
print("total average = "+str(np.mean(degrees)))

assortativity = nx.attribute_assortativity_coefficient(G, 'community')

print(f"Assortativity based on gamma: {assortativity}")
#%%



# p_in_range=np.linspace(0,1,10)
# num_of_networks=10
# ass_list_mean=np.zeros(10)
# ass_list_std=np.zeros(10)
# for j,p_in in enumerate(p_in_range):
#     temp=np.zeros(num_of_networks)
#     for i in range(num_of_networks):
#         gen_gamma(N, mu, sigma, gamma_range, gamma_dis, network)
        
#         G=nx.from_numpy_array(segregated_barabasi_albert(N, z, p_in))
#         self_loops = list(nx.selfloop_edges(G))
#         G.remove_edges_from(self_loops)
#         nx.set_node_attributes(G, value_dict, 'community')
#         assortativity = nx.attribute_assortativity_coefficient(G, 'community')
#         temp[i]=assortativity
#     ass_list_mean[j]=np.mean(temp)
#     ass_list_std[j]=np.std(temp)
        
# fig, ax = plt.subplots()
# ax.plot(p_in_range, ass_list_mean,label='Median')
# ax.fill_between(p_in_range,ass_list_mean-ass_list_std,ass_list_mean+ass_list_std, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=0,label='5-95 percentile band')
# ax.set_xlabel('p_in')
# ax.set_ylabel('Assortativity')
#%%
G=nx.barabasi_albert_graph(N, z)

degrees = [degree for node, degree in G.degree()]
clus=list(nx.clustering(G).values())


fig,ax=plt.subplots()

ax.scatter(clus,degrees,alpha=0.2)
ax.set_ylabel('Degree')
ax.set_xlabel('Local clustering')
fig, ax = plt.subplots()
bins = np.histogram_bin_edges(clus, bins='auto')
ax.hist(clus,bins=bins, alpha=0.5, label='Histogram 1')
ax.set_xlabel('Local clustering')
ax.set_ylabel('Frequency')
r=np.corrcoef(clus,degrees)[1,0]
print(r)