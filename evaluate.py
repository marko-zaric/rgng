# coding: utf-8

from re import A
from gng import GrowingNeuralGas
from rgng_matlab import RobustGrowingNeuralGas
from sklearn import datasets, metrics
import networkx as nx
import numpy as np

__authors__ = 'Adrien Guille'
__email__ = 'adrien.guille@univ-lyon2.fr'


def evaluate_on_digits():
    digits = datasets.load_digits()
    data = digits.data
    target = digits.target
    gng = GrowingNeuralGas(data)
    gng.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=100, a=0.5, d=0.995, passes=5, plot_evolution=False)
    clustered_data = gng.cluster_data()
    print('Found %d clusters.' % nx.number_connected_components(gng.network))
    target_infered = []
    for observation, cluster in clustered_data:
        target_infered.append(cluster)
    homogeneity = metrics.homogeneity_score(target, target_infered)
    print(homogeneity)
    gng.plot_clusters(gng.reduce_dimension(gng.cluster_data()))

def evaluate_on_circles():
    np.random.seed(0)
    n_samples = 1500
    #X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    X, y = datasets.make_blobs(n_samples=n_samples, centers=3, cluster_std=0.1, random_state=0)
    data = X
    target = y
    rgng = RobustGrowingNeuralGas(input_data=data, max_number_of_nodes=8, real_num_clusters=3)
    #gng = GrowingNeuralGas(data)
    #gng.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=100, a=0.5, d=0.995, passes=5, plot_evolution=False)
    resulting_centers, actual_center, mdl_values = rgng.fit_network(a_max=100, d=0.995, passes=15, plot_evolution=True)
    print(resulting_centers)
    clustered_data = rgng.cluster_data()
    #print('Found %d clusters.' % nx.number_connected_components(gng.network))
    print('Found %d clusters.' % len(resulting_centers))
    target_infered = []
    for observation, cluster in clustered_data:
        target_infered.append(cluster)
    homogeneity = metrics.homogeneity_score(target, target_infered)
    print(homogeneity)
    #rgng.plot_clusters(rgng.reduce_dimension(rgng.cluster_data()))

if __name__ == '__main__':
    evaluate_on_circles()
