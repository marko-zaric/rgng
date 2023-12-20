import scipy.io
from rgng_matlab import RobustGrowingNeuralGas
from sklearn import datasets, metrics
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy

noise_num = 0
prenode = 10

D = scipy.io.loadmat('D1data.mat')

TXoutlier84 = D["TXoutlier84"]
inputData = D["X"]
INPUT = copy.deepcopy(D["X"])
Vd1 = D["Vd1"]
mean_var = D["mean_var"]


inicenter= Vd1[:,:,0]
realcenter = mean_var[0:2,:]

X = inputData[:,:2]
y = inputData[:,2:].ravel()

num_class_labels = len(set(y))

print("Initialize Centers")
print(inicenter)
print("Real Centers")
print(realcenter)

colors = ["yellow", "green", "blue", "purple", "red"]
MAXNUMBEROFNODES = 10

if __name__ == '__main__':
    benchmark = []
    #np.random.seed(1)
    # 
    rgng = RobustGrowingNeuralGas(input_data=X, max_number_of_nodes=MAXNUMBEROFNODES, real_num_clusters=num_class_labels) #, center=inicenter)
    resulting_centers, actual_center, mdl_values = rgng.fit_network(a_max=100)
    print("Print Clusters found: ", len(resulting_centers.T[0]))
    #plt.scatter(INPUT.T[0], INPUT.T[1], c=INPUT.T[2], cmap=matplotlib.colors.ListedColormap(colors), s=10)
    plt.scatter(realcenter[0].T, realcenter[1].T,c="red", marker="^")
    plt.scatter(resulting_centers.T[0], resulting_centers.T[1],c="black", marker="s")
    plt.title("Cluster Center Difference")
    plt.show()
    print(mdl_values)
    plt.plot(np.arange(2, len(mdl_values)+2),mdl_values)
    plt.show()

    for key in rgng.optimal_receptive_field.keys():
        vectors = np.array(rgng.optimal_receptive_field[key]['input'])
        plt.scatter(resulting_centers.T[0], resulting_centers.T[1],c="black", marker="s")
        plt.scatter(vectors.T[0], vectors.T[1], c=colors[key], s=10)
    plt.show()
