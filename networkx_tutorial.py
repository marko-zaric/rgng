# coding: utf-8

# import numpy as np
# from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
# from sklearn import decomposition

__authors__ = 'Erwan Renaudo'
__email__ = 'erwan.renaudo@gmail.com'

'''
Testing networkx
'''

G = nx.Graph()

# nodes
G.add_node(1)
G.add_nodes_from([2,3])
G.add_nodes_from([(4,{"type": "fire"}),(5,{"type": "glass"})])
G.add_nodes_from([(6,{"type": "water"}),(7,{"type": "air"})])
G.add_nodes_from([(8,{"type": "fire"}),(9,{"type": "glass"})])
G.add_nodes_from([(10,{"type": "water"}),(11,{"type": "glass"})])
G.add_nodes_from([(12,{"type": "glass"}),(13,{"type": "glass"})])

# edges
G.add_edge(1,2)
e = (2,3)
G.add_edge(*e)
G.add_edges_from([(4,5), (11,12), (3,5, {"weight": 0.1574})])
G.add_edges_from([(3,5), (1,3), (1,5, {"weight": 0.984})])
G.add_edges_from([(6,5), (7,11), (12,4, {"weight": 0.984})])
G.add_edges_from([(13,2), (10,8), (12,10, {"weight": 0.984})])
G.add_edges_from([(7,9), (10,6), (13,3, {"weight": 0.984})])

print("Nodes: %d, Edges: %d" % (G.number_of_nodes(), G.number_of_edges()))
print(G.nodes, G.edges)

#subax1 = plt.subplot(1,1,1)
#subax1 = plt.subplot(2,1,1)
subax1 = plt.subplot(2,2,1)
nx.draw(G, with_labels=True)

subax2 = plt.subplot(2,2,2)
nx.draw_shell(G, with_labels=True)

subax3 = plt.subplot(2,2,3)
nx.draw_circular(G, with_labels=True)

subax4 = plt.subplot(2,2,4)
nx.draw_spectral(G, with_labels=True)

plt.show()