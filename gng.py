# coding: utf-8

import numpy as np
from scipy import spatial
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import decomposition
import statistics

__authors__ = 'Adrien Guille'
__email__ = 'adrien.guille@univ-lyon2.fr'

'''
Simple implementation of the Growing Neural Gas algorithm, based on:
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
'''

#test for exact equality
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


class GrowingNeuralGas:

    def __init__(self, input_data):
        self.network = None
        self.data = input_data
        self.eta = 0.0001
        self.units_created = 0
        self.inputted_vectors = []
        self.outliers = []
        plt.style.use('ggplot')

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.network.nodes(data=True):
            vector = attributes['vector']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def prune_connections(self, a_max):
        nodes_to_remove = []
        for u, v, attributes in self.network.edges(data=True):
            if attributes['age'] > a_max:
                nodes_to_remove.append((u, v))
        for u, v in nodes_to_remove:
            self.network.remove_edge(u, v)

        nodes_to_remove = []
        for u in self.network.nodes():
            if self.network.degree(u) == 0:
                nodes_to_remove.append(u)
        for u in nodes_to_remove:
            self.network.remove_node(u)

    def fit_network(self, e_b, e_n, a_max, l, a, d, passes=1, plot_evolution=False):
        # logging variables
        accumulated_local_error = []
        global_error = []
        network_order = []
        network_size = []
        total_units = []
        self.units_created = 0
        # 0. start with two units a and b at random position w_a and w_b
        w_a = [np.random.uniform(-2, 2) for _ in range(np.shape(self.data)[1])]
        w_b = [np.random.uniform(-2, 2) for _ in range(np.shape(self.data)[1])]
        self.network = nx.Graph()
        self.network.add_node(self.units_created, vector=w_a, error=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0)
        self.units_created += 1
        # 1. iterate through the data
        sequence = 0
        self.data_range = np.max(self.data, axis=0) - np.min(self.data, axis=0)
        for p in range(passes):
            self.receptive_field = {}
            print('   Pass #%d' % (p + 1))
            np.random.shuffle(self.data)
            steps = 0
            for observation in self.data:
                if arreq_in_list(observation, self.inputted_vectors):
                    pass
                else:
                    self.inputted_vectors.append(observation)
                # 2. find the nearest unit s_1 and the second nearest unit s_2
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]

                if s_1 not in self.receptive_field.keys():
                    self.receptive_field[s_1] = {'input': [observation]}
                else:
                    self.receptive_field[s_1]['input'].append(observation)

                # 3. increment the age of all edges emanating from s_1
                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age']+1)
                # 4. add the squared distance between the observation and the nearest unit in input space
                self.network.nodes[s_1]['error'] += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
                # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
                #    e_b and e_n, respectively, of the total distance
                #update_w_s_1 = self.e_b * \
                update_w_s_1 = e_b * \
                    (np.subtract(observation,
                                 self.network.nodes[s_1]['vector']))
                self.network.nodes[s_1]['vector'] = np.add(
                    self.network.nodes[s_1]['vector'], update_w_s_1)

                for neighbor in self.network.neighbors(s_1):
                    #update_w_s_n = self.e_n * \
                    update_w_s_n = e_n * \
                        (np.subtract(observation,
                                     self.network.nodes[neighbor]['vector']))
                    self.network.nodes[neighbor]['vector'] = np.add(
                        self.network.nodes[neighbor]['vector'], update_w_s_n)
                # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
                #    if such an edge doesn't exist, create it
                self.network.add_edge(s_1, s_2, age=0)
                # 7. remove edges with an age larger than a_max
                #    if this results in units having no emanating edges, remove them as well
                self.prune_connections(a_max)
                # 8. if the number of steps so far is an integer multiple of parameter l, insert a new unit
                steps += 1
                if steps % l == 0:
                    if plot_evolution:
                        self.plot_network('visualization_gng/sequence/' + str(sequence) + '.png')
                    sequence += 1
                    # 8.a determine the unit q with the maximum accumulated error
                    q = 0
                    error_max = 0
                    for u in self.network.nodes():
                        if self.network.nodes[u]['error'] > error_max:
                            error_max = self.network.nodes[u]['error']
                            q = u
                    # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
                    f = -1
                    largest_error = -1
                    for u in self.network.neighbors(q):
                        if self.network.nodes[u]['error'] > largest_error:
                            largest_error = self.network.nodes[u]['error']
                            f = u
                    w_r = 0.5 * (np.add(self.network.nodes[q]['vector'], self.network.nodes[f]['vector']))
                    r = self.units_created
                    self.units_created += 1
                    # 8.c insert edges connecting the new unit r with q and f
                    #     remove the original edge between q and f
                    self.network.add_node(r, vector=w_r, error=0)
                    self.network.add_edge(r, q, age=0)
                    self.network.add_edge(r, f, age=0)
                    self.network.remove_edge(q, f)
                    # 8.d decrease the error variables of q and f by multiplying them with a
                    #     initialize the error variable of r with the new value of the error variable of q
                    self.network.nodes[q]['error'] *= a
                    self.network.nodes[f]['error'] *= a
                    self.network.nodes[r]['error'] = self.network.nodes[q]['error']
                # 9. decrease all error variables by multiplying them with a constant d
                error = 0
                for u in self.network.nodes():
                    error += self.network.nodes[u]['error']
                accumulated_local_error.append(error)
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                for u in self.network.nodes():
                    self.network.nodes[u]['error'] *= d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)
            global_error.append(self.compute_global_error())
            print("MDL: ", self.compute_MDL())
            

        plt.clf()
        plt.title('Accumulated local error')
        plt.xlabel('iterations')
        plt.plot(range(len(accumulated_local_error)), accumulated_local_error)
        plt.savefig('visualization_gng/accumulated_local_error.png')
        plt.clf()
        plt.title('Global error')
        plt.xlabel('passes')
        plt.plot(range(len(global_error)), global_error)
        plt.savefig('visualization_gng/global_error.png')
        plt.clf()
        plt.title('Neural network properties')
        plt.plot(range(len(network_order)), network_order, label='Network order')
        plt.plot(range(len(network_size)), network_size, label='Network size')
        plt.legend()
        plt.savefig('visualization_gng/network_properties.png')

        return global_error, accumulated_local_error

    def plot_network(self, file_path):
        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        node_pos = {}
        for u in self.network.nodes():
            vector = self.network.nodes[u]['vector']
            node_pos[u] = (vector[0], vector[1])
        nx.draw(self.network, pos=node_pos)
        plt.draw()
        plt.savefig(file_path)

    def number_of_clusters(self):
        return nx.number_connected_components(self.network)

    def cluster_data(self):
        unit_to_cluster = np.zeros(self.units_created)
        cluster = 0
        for c in nx.connected_components(self.network):
            for unit in c:
                unit_to_cluster[unit] = cluster
            cluster += 1
        clustered_data = []
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s = nearest_units[0]
            clustered_data.append((observation, unit_to_cluster[s]))
        return clustered_data

    def reduce_dimension(self, clustered_data):
        transformed_clustered_data = []
        svd = decomposition.PCA(n_components=2)
        transformed_observations = svd.fit_transform(self.data)
        for i in range(len(clustered_data)):
            transformed_clustered_data.append((transformed_observations[i], clustered_data[i][1]))
        return transformed_clustered_data

    def compute_MDL(self):
        # evaluate outlier set (Delta L, (13))
        # For x in w receptive field
        c = len(self.network.nodes()) # number of prototypes
        K = np.log2( (self.data_range[0] - self.data_range[1]) / self.eta) # number of bits needed to encode a single data vector
        N = np.shape(self.data)[0] # number of input vectors
        d = np.shape(self.data)[1] # dimension of input vectors
        num_outliers = 0
        for observation in self.inputted_vectors:
            nearest_units = self.find_nearest_units(observation)
            proto = nearest_units[0]
            try:
                #print(self.receptive_field[proto]["input"])
                S_i = self.receptive_field[proto]["input"] # receptive field
                if observation in S_i and len(S_i) == 1:
                    phi_Si = 1 
                else:
                    phi_Si = 0
            except:
                phi_Si = 0
            # dectect outliers with eq (13)
            # deltaL = ((N - 1) * np.log2(c - phi_Si) + K) \
            # - (N * np.log2(c) + np.sum( [ np.max([ np.log2(np.abs(observation[k] - self.network.nodes[proto]['vector'][k]) / self.eta ),1]) for k in range(d)] ) ) \
            # - phi_Si * K
            new_outliers = []
            deltaL = K + (len(self.inputted_vectors)-len(self.outliers))*(np.log2(c-1) - np.log2(c)) \
                    + np.sum( [ np.max([ np.log2(np.abs(observation[k] - self.network.nodes[proto]['vector'][k]) / self.eta ),1]) for k in range(d)] )

          
            if deltaL < 0:
                new_outliers.append(observation)
                num_outliers += 1
        self.outliers = new_outliers

        MDL = c*K + N * np.log2(c) + len(self.outliers)*K 
        sum_mdl_term = 0
        for prototype in self.network.nodes():
            try:
                S_i = self.receptive_field[prototype]["input"]
            except:
                continue

            for x in S_i:
                sum_mdl_term += np.sum( [ np.max([ np.log2(np.abs(x[k] - self.network.nodes[prototype]['vector'][k]) / self.eta ),1]) for k in range(d)] )

        MDL += 1.3 * sum_mdl_term
        print("Number of outliers: ", num_outliers)

        return MDL

    def plot_clusters(self, clustered_data):
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization_gng/clusters.png')

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
        return global_error

    def h_mean(self, a):
        return statistics.harmonic_mean(a)

