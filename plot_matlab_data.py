import scipy.io
from rgng_matlab import RobustGrowingNeuralGas
from sklearn import datasets, metrics
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

noise_num = 0
prenode = 10

D = scipy.io.loadmat('D1data.mat')
#.
TXoutlier84 = D["TXoutlier84"]
inputData = D["X"]
Vd1 = D["Vd1"]
mean_var = D["mean_var"]


inicenter= Vd1[:,:,0]
realcenter = mean_var[0:2,:]

X = inputData[:,:2]
y = inputData[:,2:].ravel()

num_class_labels = len(set(y))

print("Vd1(:,:,1)")
print(inicenter)
print("mean_var(1:2,:)")
print(realcenter)

colors = ["yellow", "green", "blue", "purple", "red"]


if __name__ == '__main__':
    
    plt.scatter(inputData.T[0], inputData.T[1], c=inputData.T[2], cmap=matplotlib.colors.ListedColormap(colors), s=10)
    plt.scatter(realcenter[0].T, realcenter[1].T,c="black", marker="^")
    plt.show()