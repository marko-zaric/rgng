# coding: utf-8

from gng import GrowingNeuralGas
from rgng_play_around import RobustGrowingNeuralGas
#from rgng import RobustGrowingNeuralGas
from sklearn import datasets, metrics
import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt

__authors__ = 'Adrien Guille, Marko Zaric'
__email__ = 'adrien.guille@univ-lyon2.fr, marko.zaric@uibk.ac.at'

def evaluate(model):
    global_error, accomulated_error = model.fit_network(e_b=0.05, e_n=0.006, a_max=8, l=300, a=0.5, d=0.995, passes=15, plot_evolution=True)
    clustered_data = model.cluster_data()
    print('Found %d clusters.' % nx.number_connected_components(model.network))
    target_infered = []
    for observation, cluster in clustered_data:
        target_infered.append(cluster)
    homogeneity = metrics.homogeneity_score(target, target_infered)
    print(homogeneity)
    model.plot_clusters(model.reduce_dimension(model.cluster_data()))
    return global_error, accomulated_error, homogeneity

if __name__ == '__main__':
    benchmark = []
    np.random.seed(0)
    n_samples = 1500
    X, y = datasets.make_blobs(n_samples=n_samples, centers=3, random_state=0, center_box=(-30,30))
    #X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    data = X
    target = y
    benchmark.append(time.time())
    rgng = RobustGrowingNeuralGas(data)
    global_error_rgng, accomulated_error_rgng, homogeneity_rgng = evaluate(rgng)
    benchmark.append(time.time())
    gng = GrowingNeuralGas(data)
    global_error_gng, accomulated_error_gng, homogeneity_gng = evaluate(gng)
    benchmark.append(time.time())

    print("GNG time: ", benchmark[1]-benchmark[0])
    print("GNG homogenity: ", homogeneity_gng)
    print("RGNG time: ", benchmark[2]-benchmark[1])
    print("RGNG homogenity: ", homogeneity_rgng)
    plt.clf()
    plt.title('Accumulated local error')
    plt.xlabel('iterations')
    plt.plot(range(len(accomulated_error_gng)-3000), accomulated_error_gng[3000:], label='GNG')
    plt.plot(range(len(accomulated_error_rgng)-3000), accomulated_error_rgng[3000:], label='RGNG')
    plt.legend()
    plt.show()
    plt.title('Global error')
    plt.xlabel('passes')
    plt.plot(range(len(global_error_gng)-2), global_error_gng[2:], label='GNG')
    plt.plot(range(len(global_error_rgng)-2), global_error_rgng[2:], label='RGNG')
    plt.legend()
    plt.show()

    



