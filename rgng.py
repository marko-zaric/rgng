# coding: utf-8

import numpy as np
from scipy import spatial
from scipy import stats
import networkx as nx
from RGNG_network import RGNG_Graph
import matplotlib.pyplot as plt
from sklearn import decomposition
import statistics

__authors__ = 'Adrien Guille, Erwan Renaudo'
__email__ = 'adrien.guille@univ-lyon2.fr, erwan.renaudo@uibk.ac.at'

'''
Implementation of Robust Growing Neural Gas based on Adrien Guille's GNG code and
A Growing Neural Gas Network Learns Topologies. B. Fritzke, Advances in Neural
Information Processing Systems 7, 1995.
'''


class RobustGrowingNeuralGas:

    def __init__(self, input_data):
        self.network = None
        self.data = input_data
        self.units_created = 0
        self.beta_integral_mul = 2
        self.fsigma_weakening = 0.1
        self.e_bi = 0.1
        self.e_bf = 0.01
        self.e_ni = 0.005
        self.e_nf = 0.0005
        self.eta = 0.0001
        self.prenode_rank = []
        self.receptive_field = {}
        self.max_nodes = len(input_data)
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
        self.network = RGNG_Graph() # nx.Graph()
        self.network.add_node(self.units_created, vector=w_a, error=0, e_b=0, e_n=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0, e_b=0, e_n=0)
        self.units_created += 1
        # adaptive rate initial update
        self.prenode_rank = [self.units_created - 1 - u for u in np.arange(self.units_created)]
        #for n in self.network.nodes():
        #    self.receptive_field[n] = {'input': []}
        # print("network ", self.network)
        # print("network nodes ", self.network.nodes())
        # print("network ranks", self.prenode_rank)
        # print("receptive fields", self.receptive_field)
        # print("DATA SIZE", len(self.data))
        # print('DATA SHAPE', np.shape(self.data))
        # print('DATA SHAPE', np.shape(self.data)[1])

        # 1. iterate through the data
        sequence = 0
        #print("#DEBUG: HARDCODED NB OF PASSES")
        #passes = 2

        self.data_range = np.max(self.data, axis=0) - np.min(self.data, axis=0)
        print(self.data_range)
        print(len(np.minimum(self.data, self.data)))

        for p in range(passes):
            # reset the receptive fields
            self.receptive_field = {}

            print('   Pass #%d' % (p + 1))

            # Compute initial restricting distance for each prototype (as harmonic mean of error)
            #print("1.5, initial restr dist computation")
            #print([[np.linalg.norm( obs - n_[1]['vector'] ) for obs in self.data] for n_ in self.network.nodes.items()])
            self.d_restr = { n_[0]: self.h_mean([np.linalg.norm( obs - n_[1]['vector'] ) for obs in self.data]) for n_ in self.network.nodes.items() }

            # Compute learning rates here ?
            # (b)2) adaptive learning rate
            for node in self.network.nodes():
                self.network.nodes[node]['e_b'] = self.e_bi * pow((self.e_bf / self.e_bi ), self.network.nodes[node]["prenode_ranking"] / self.max_nodes)
                self.network.nodes[node]['e_n'] = self.e_ni * pow((self.e_nf / self.e_ni ), self.network.nodes[node]["prenode_ranking"] / self.max_nodes)


            np.random.shuffle(self.data)
            steps = 0
            # QinS2004 Table 2(b)
            for observation in self.data:
                global_iter = steps + p * len(self.data)
                # store restricting dist t-1:
                self.d_restr_prev = self.d_restr
                
                # 2. find the nearest unit s_1 and the second nearest unit s_2
                nearest_units = self.find_nearest_units(observation)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]

                #print("WINNER", s_1)
                #QinS2004 RGNG
                # Update receptive field of winner (optimal cluster size)
                if s_1 not in self.receptive_field.keys():
                    #print("NEW WINNER")
                    self.receptive_field[s_1] = {'input': [observation]}
                else:
                    #print("update receptive field f ", s_1, " with ",observation)
                    #print("pre", len(self.receptive_field[s_1]['input']))
                    self.receptive_field[s_1]['input'].append(observation)
                    #print("post", len(self.receptive_field[s_1]['input']))
                    #print(np.any(observation in self.receptive_field[s_1]['input']))
                    #print(self.receptive_field[s_1]['input'])
                    #self.receptive_field[s_1]['input'] = set(self.receptive_field[s_1]['input'].append(observation))

                
                #print("receptive fields", self.receptive_field[s_1])
                #print("receptive fields", self.receptive_field)
                # Compute average distance between s1 and its neighbours for eq. (10)
                #print("data", observation)
                #print("NEIGHBORS",[nb for nb in self.network.neighbors(s_1)])
                #print("NEIGHBORS", len(list(self.network.neighbors(s_1))))
                avg_neighbor_dist = 0
                if len(list(self.network.neighbors(s_1))) > 0:
                    avg_neighbor_dist = sum([spatial.distance.euclidean(s_1, nb) for nb in self.network.neighbors(s_1)])/len(list(self.network.neighbors(s_1)))
                # else no neighbour, distance ws1 - wi does not exist, value is 0

                # 3. increment the age of all edges emanating from s_1
                for u, v, attributes in self.network.edges(data=True, nbunch=[s_1]):
                    self.network.add_edge(u, v, age=attributes['age']+1)
                # 4. add the squared distance between the observation and the nearest unit in input space
                self.network.nodes[s_1]['error'] += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
                # =================================================
                # 5. move s_1 and its direct topological neighbors towards the observation by the fractions
                # /!\ RGNG outlier resistant strategy
                # 5.1 Knowing s_1, update d_s_1
                self.d_restr[s_1] = self.update_restricting_dist(self.network.nodes[s_1]['vector'], observation, self.d_restr_prev[s_1])

                s1_error = np.subtract(observation, self.network.nodes[s_1]['vector'])
                normed_s1_error = np.subtract(observation, self.network.nodes[s_1]['vector'])
                normed_s1_error /= np.linalg.norm(normed_s1_error)

                # 5.2
                # ------
                # adaptive learning rate:
                # print("s_1",s_1)
                # print(self.network.nodes(data=True))
                e_b = self.e_bi * pow((self.e_bf / self.e_bi ), self.prenode_rank[s_1] / self.max_nodes)
                # print("s1, eb: ",s_1,e_b)
                # print("s1, eb2: ", s_1, self.network.nodes[s_1]['e_b'])
                # print(self.prenode_rank[s_1])
                # print(self.network.nodes[s_1]["prenode_ranking"])
                # prototype update
                update_w_s_1 = self.network.nodes[s_1]['e_b'] * self.sigma_modulation(self.network.nodes[s_1]['vector'], observation, self.d_restr_prev[s_1]) * normed_s1_error

                self.network.nodes[s_1]['vector'] = np.add(
                    self.network.nodes[s_1]['vector'], update_w_s_1)

                # ------
                # TODO:
                for neighbor in self.network.neighbors(s_1):
                    # QinS2004 eq (10), repulsive force direction
                    s1_to_neighbor = np.subtract(neighbor, self.network.nodes[s_1]['vector'])
                    s1_to_neighbor /= np.linalg.norm(s1_to_neighbor)

                    obs_to_neighbor = np.subtract(observation, neighbor)
                    obs_to_neighbor /= np.linalg.norm(obs_to_neighbor)
                    # ------
                    # adaptive rate
                    e_n = self.e_ni * pow((self.e_nf / self.e_ni ), self.prenode_rank[neighbor] / self.max_nodes)
                    # print("n, en: ",neighbor,e_n)
                    # print("n, en2: ", neighbor, self.network.nodes[neighbor]['e_n'])
                    # print(self.prenode_rank[neighbor])
                    # print(self.network.nodes[neighbor]["prenode_ranking"])
                    # ------

                    self.d_restr[neighbor] = self.update_restricting_dist(self.network.nodes[neighbor]['vector'], observation, self.d_restr_prev[neighbor])
                    update_w_s_n = self.network.nodes[neighbor]['e_n'] * self.sigma_modulation(self.network.nodes[neighbor]['vector'], observation, self.d_restr_prev[neighbor]) * obs_to_neighbor + np.exp( - spatial.distance.euclidean(s_1, neighbor) / self.fsigma_weakening) * self.beta_integral_mul * avg_neighbor_dist * s1_to_neighbor

                    self.network.nodes[neighbor]['vector'] = np.add(
                        self.network.nodes[neighbor]['vector'], update_w_s_n)

                # =================================================
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
                        self.plot_network('visualization/sequence/' + str(sequence) + '.png')
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
                    self.network.add_node(r, vector=w_r, error=0, e_b=0, e_n=0)

                    for node in self.network.nodes():
                        self.network.nodes[node]['e_b'] = self.e_bi * pow((self.e_bf / self.e_bi ), self.network.nodes[node]["prenode_ranking"] / self.max_nodes)
                        self.network.nodes[node]['e_n'] = self.e_ni * pow((self.e_nf / self.e_ni ), self.network.nodes[node]["prenode_ranking"] / self.max_nodes)

                    self.network.add_edge(r, q, age=0)
                    self.network.add_edge(r, f, age=0)
                    self.network.remove_edge(q, f)

                    # update the restricting distance to include the new node
                    self.d_restr[r] = self.h_mean([np.linalg.norm( obs - self.network.nodes[r]['vector'] ) for obs in self.data])
                    
                    # update the prenode_rank to age older nodes and include the new node
                    self.prenode_rank = [x_ + 1 for x_ in self.prenode_rank]
                    self.prenode_rank.append(0)

                    # 8.d decrease the error variables of q and f by multiplying them with a
                    # initialize the error variable of r with the new value of the error variable of q
                    self.network.nodes[q]['error'] *= a
                    self.network.nodes[f]['error'] *= a
                    self.network.nodes[r]['error'] = self.network.nodes[q]['error']
                # 9. decrease all error variables by multiplying them with a constant d
                error = 0
                for u in self.network.nodes():
                    error += self.network.nodes[u]['error']

                # 10. Update MDL
                #print("Delta L: ",self.compute_MDL(s_1, observation))

                accumulated_local_error.append(error)
                network_order.append(self.network.order())
                network_size.append(self.network.size())
                total_units.append(self.units_created)
                for u in self.network.nodes():
                    self.network.nodes[u]['error'] *= d
                    if self.network.degree(nbunch=[u]) == 0:
                        print(u)
            global_error.append(self.compute_global_error())

        
        #print("receptive fields", len(self.receptive_field))
        for key, rf in self.receptive_field.items():
            print(key, len(rf['input']))
        print("total sample considered:", np.sum([len(rf['input']) for k, rf in self.receptive_field.items()]))

        print("network nodes ", self.network.nodes())
        print(self.prenode_rank)
        plt.clf()
        plt.title('Accumulated local error')
        plt.xlabel('iterations')
        plt.plot(range(len(accumulated_local_error)), accumulated_local_error)
        plt.savefig('visualization/accumulated_local_error.png')
        plt.clf()
        plt.title('Global error')
        plt.xlabel('passes')
        plt.plot(range(len(global_error)), global_error)
        plt.savefig('visualization/global_error.png')
        plt.clf()
        plt.title('Neural network properties')
        plt.plot(range(len(network_order)), network_order, label='Network order')
        plt.plot(range(len(network_size)), network_size, label='Network size')
        plt.legend()
        plt.savefig('visualization/network_properties.png')

        return global_error, accumulated_local_error

    def compute_MDL(self, proto, observation):
        # evaluate outlier set (Delta L, (13))
        # For x in w receptive field
        c = len(self.network.nodes())
        K = np.log2( self.data_range / self.eta)
        N = np.shape(self.data)[0]
        d = np.shape(self.data)[1]
        S_i = self.receptive_field[proto]
        phi_Si = 1 if len(S_i) == 1 else 0

        print('c, K, N, d, S_i, phi_Si')
        print(c, K, N, d, len(S_i), phi_Si)
        print(proto)
        print(self.network.nodes[proto]['vector'])
        print(observation)
        print("Obs len",len(observation))
        print(observation[0])
        print(observation[1])
        print(np.arange(d))
        print(self.network.nodes[proto]['vector'][0])
        print(self.network.nodes[proto]['vector'][1])
        print('E------')
        #print([np.log2( np.abs(observation[k] - self.network.nodes[proto]['vector'][k]))  for k in np.arange(d)])
        print([np.log2( np.linalg.norm(observation[k] - self.network.nodes[proto]['vector'][k]))  for k in np.arange(d)])

        # np.linalg.norm(obs-proto) ?
        #TODO
        print("Compute MDL -- Delta L")
        # l1 ok
        # l2
        deltaL = ((N - 1) * np.log2(c - phi_Si) + K) \
        #- (N * np.log2(c) + np.sum( [ np.max( np.log2(np.linalg.norm((observation[k] - self.network.nodes[proto]['vector'][k]) / self.eta )),1) for k in np.arange(d)] ) )
        #- phi_Si * K
        # mod L(I,W) = c*K + N log2 c
        # error L(I,W) = 
        # mod L(0) = card(O) * K

        return deltaL

    # ==================================
    # proto, obs, d(-1)
    #(self.network.nodes[s_1]['vector'], observation, self.d_restr_prev[s_1])
    def sigma_modulation(self, proto, observation, prev):
        # pseudo algo
        current_error = np.linalg.norm(observation - proto)
        if current_error < prev:
            return current_error
        else:
            return prev

    def update_restricting_dist(self, proto, observation, prev):
        current_error = np.linalg.norm(observation - proto)
        if current_error < prev:
            # arithmetic mean
            return 0.5 * (prev + current_error)
        else:
            # harmonic mean
            return statistics.harmonic_mean([prev, current_error])

    # array a
    def h_mean(self, a):
        return statistics.harmonic_mean(a)

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

    def plot_clusters(self, clustered_data):
        number_of_clusters = nx.number_connected_components(self.network)
        plt.clf()
        plt.title('Cluster affectation')
        color = ['r', 'b', 'g', 'k', 'm', 'r', 'b', 'g', 'k', 'm']
        for i in range(number_of_clusters):
            print(i)
            observations = [observation for observation, s in clustered_data if s == i]
            if len(observations) > 0:
                observations = np.array(observations)
                plt.scatter(observations[:, 0], observations[:, 1], color=color[i], label='cluster #'+str(i))
        plt.legend()
        plt.savefig('visualization/clusters.png')

    def compute_global_error(self):
        global_error = 0
        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
        return global_error


