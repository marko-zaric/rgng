#!/usr/bin/env python
# coding: utf-8

from rgng_play_around import RobustGrowingNeuralGas
from gng import GrowingNeuralGas
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import shutil

__authors__ = 'Adrien Guille'
__email__ = 'adrien.guille@univ-lyon2.fr'

if __name__ == '__main__':
    if os.path.exists('visualization/sequence'):
        shutil.rmtree('visualization/sequence')
    os.makedirs('visualization/sequence')
    n_samples = 2000
    #n_samples = 2000
    dataset_type = 'blobs'
    data = None
    noise = np.random.uniform(-1.0,1.0, size=(n_samples,2))
    print('Preparing data...')
    if dataset_type == 'blobs':
        data = datasets.make_blobs(n_samples=n_samples, centers=8, random_state=0)
    elif dataset_type == 'moons':
        data = datasets.make_moons(n_samples=n_samples, noise=.05)
    elif dataset_type == 'circles':
        data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.075)
    data = StandardScaler().fit_transform(data[0]) #+ noise
    print('Done.')
    print('Fitting neural network ONLINE')
    n_sample_per_batch = 1000
    for i in np.arange(1,n_samples/n_sample_per_batch, dtype=int):
        print('Data phase %d' % i)
        gng = RobustGrowingNeuralGas(data[0:i*n_sample_per_batch])
        gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=8, plot_evolution=True)
        print('Found %d clusters.' % gng.number_of_clusters())
        gng.plot_clusters(gng.cluster_data())
        #wait = raw_input("Press to go to next phase")
        print("HM test: ", gng.h_mean([0.5,5,5,7,8,2,1]))
        wait = input("Press to go to next phase")