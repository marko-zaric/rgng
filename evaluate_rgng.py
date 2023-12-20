import scipy.io
from rgng_matlab import RobustGrowingNeuralGas
from sklearn import datasets, metrics
from sklearn.preprocessing import StandardScaler
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy
from sklearn import datasets, metrics
colors = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
noise_num = 0
prenode = 10

n_samples = 1500
X, y = datasets.make_moons(n_samples=1500)#datasets.make_blobs(n_samples=n_samples, centers=3, cluster_std=0.1, random_state=0)

scaler = StandardScaler()
X = scaler.fit_transform(X)
INPUT = copy.deepcopy(X)

c = [colors[i] for i in y]

target = y
num_class_labels = len(set(target))




MAXNUMBEROFNODES = 20

if __name__ == '__main__':
    benchmark = []
    #np.random.seed(1)
    # 
    rgng = RobustGrowingNeuralGas(input_data=X, max_number_of_nodes=MAXNUMBEROFNODES, real_num_clusters=num_class_labels) #, center=inicenter)
    resulting_centers, actual_center, mdl_values = rgng.fit_network(a_max=10, passes=20)
    print(resulting_centers)
    print(mdl_values)
    plt.plot(np.arange(2, len(mdl_values)+2),mdl_values)
    plt.show()
    plt.scatter(INPUT.T[0], INPUT.T[1], c=c, cmap=matplotlib.colors.ListedColormap(colors), s=10)
    node_pos = {}
    for u in rgng.optimal_network.nodes():
        vector = rgng.optimal_network.nodes[u]['vector']
        node_pos[u] = (vector[0], vector[1])
    nx.draw(rgng.optimal_network, pos=node_pos)
    plt.draw()
    
    plt.show()

    
    # for key in rgng.optimal_receptive_field.keys():
    #     vectors = np.array(rgng.optimal_receptive_field[key]['input'])
    #     print("Node: ", key)
    #     print(vectors.shape)
    #     plt.scatter(vectors.T[0], vectors.T[1], c=colors[key], s=10)
    # plt.scatter(resulting_centers.T[0], resulting_centers.T[1],c="black", marker="s")
    # plt.show()
