# coding: utf-8

import numpy as np
from RGNG_network import RGNG_Graph

units_created = 0
w_a = np.array([1,2])
w_b = np.array([2,1])

network = RGNG_Graph()
network.add_node(units_created, vector=w_a, error=0)
units_created += 1
network.add_node(units_created, vector=w_b, error=0)
units_created += 1

print(network.nodes())
for n in network.nodes():
    print(network.nodes[n]['prenode_ranking'])
network.add_node(units_created, vector=w_b, error=0)
units_created += 1

print(network.nodes())
for n in network.nodes():
    print(network.nodes[n]['prenode_ranking'])