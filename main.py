import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import spearmanr
import pandas as pd
import tqdm

from graph_specs import graph_specs
from update_node import update_node_category, update_node_normal
from formula import *
from utilities import *

class NetworkSimulation(object):
	""" 
	NetworkSimulation contains a generic network simulation class that can create network graph
	of different type, upgrate and simulate the dynamic changes of network over time.
	"""
	def __init__(self, graph_type = 'random', 	# Options: 'random', 'dyadic_error', 'collective_error', 'category', 'ddd'

				 self_fulfilling = False,		# Whether the actor follows the self-fulfilling rule, detailed on self.phi

				 simulation_times = 100, 		# Num of times repeating the simulaiton with the same params

				 iteration_times = 20, 			# Num of times to update the network in each simualtion

				 actor_num = 30, 				# Num of actors participating in the simulation

				 nominal_num = 2,				# Num of nominal values in the given list

				 err_level = 1, 				# Error level for error graph

				 w = 0.2, 						# Social influence: to which degree actors will recognize each other with
				 								# their social reputation

				 s = 1, 						# Symmetry: to which degree actors follow the reciprocity principle

				 phi = 0.5, 					# Parameter in self-fulfulling situation: to which degree actors will  
												# based their behaviours onhow they were treated in previous round

				 h = 1, 						# Limit set to restrict the interaction between actors

				 si = 0.5, 						# Actors' sensitivity to the difference between the given/received amount
												# of deference in dyadic encounters

				 category_num = 10,				# Num of categories
				 ):

		self.graph_type = graph_type
		self.self_fulfilling = self_fulfilling
		self.simulation_times = simulation_times
		self.iteration_times = iteration_times
		self.actor_num = actor_num
		self.nominal_num = nominal_num
		self.err_level = err_level
		self.w = w
		self.s = s
		self.phi = phi
		self.h = h
		self.si = si
		self.category_num = category_num

		if self.graph_type not in graph_specs.keys():
			raise ValueError('Please specify one of the following graphs: ' + ','.join(graph_specs.keys()))

		self.update_graph = graph_specs[self.graph_type]['update_graph']

		ini_category = np.arange(self.category_num, step = 1).tolist()
		category = [x + 1 for x in ini_category]

		self.category = category

	def graph_setup(self):
		"""
		Set up the graph with initial parameters.
		"""

		G = graph_specs[self.graph_type]['setup'](self.actor_num, self.err_level, self.nominal_num, self.category_num)

		# Calculate the status score at initialization stage
		for i in range(self.actor_num):
			G.node[i]['status'] = calculate_status(i, G)

		# Calculate the standardized quality of each node by standardizing the status score		
		for j in range(self.actor_num):
			G.node[j]['z_score'] = z_score_of_node(j, G)

		quality_rank_list = self.initial_rank_based_on_quality(G)

		return G, quality_rank_list

	def simulation(self):
		"""
		Iterate the update process multiple times to simulate the development of network.
		"""

		G, quality_rank_list = self.graph_setup()

		status_rank_list = []

		for tt in range(self.iteration_times):
			if self.graph_type == 'category':
				# Create a matrix to keep track of the choice history of each actor
				choice_history = np.zeros(shape = (30, 20), dtype = np.int8)

				for k in range(self.actor_num):
					# Update the choice history of tt round
					choice_history[k][tt] = G.node[k]['category']
					# At the beginning of each iteration, update the category of each node in advance
					G.node[k]['category'] = np.random.choice(self.category)
				if tt > 0:
					# To get the biggest similarity among all, a dictionary is created to record the similarity for all possible connection
					sim_dict = {}
					pair_list = all_pairs(self.actor_num)
					for each_pair in pair_list:
						sim_dict[each_pair] = calculate_similarity(choice_history, each_pair[0], each_pair[1])
					G, status_rank = self.update_graph(G, sim_dict, self.w, self.s, update_node_category)
					# ipdb.set_trace()
				else:
					# Do nothing at the first round
					status_rank = quality_rank_list
			else:
				G, status_rank = self.update_graph(G, self.graph_type, self.nominal_num, self.err_level, self.self_fulfilling, self.w, self.s, self.phi, self.h, self.si, update_node_normal)
			status_rank_list.append(status_rank)
		last_round_rank_list = self.last_round_rank(status_rank_list)

		correlation = spearmanr(quality_rank_list, last_round_rank_list)[0]
		return correlation

	def multiple_simulations(self):
		"""
		This function can be called directly for simulation with different parameters.
		Simulating the network development for multiple times and get the averaged result.
		"""

		cor_list = []
		for ii in tqdm.trange(self.simulation_times):
			cor = self.simulation()
			cor_list.append(cor)

		return np.mean(cor_list)

	def processing_rank_list_of_node(self, node, ini_rank_list):
		"""
		Change the rank list into the list of rank status of each node.
		Desired output format: node 0: [2, 1, 2, 3, 2, 2, 2, ... 2].
		Number in the list denotes the rank of the node in each round.
		"""

		node_rank_list = []
		# rank_list: The result of the rank for all nodes in each round, ordered from the lowest to the highest
		# This function returns the rank of each round for each node
		for rank in ini_rank_list:
			node_rank_list.append(rank.index(node)+1)
		return node_rank_list

	def rank_change_over_time(self, ini_rank_list):
		"""
		Plot the changes of rank for each node over time.
		"""
		plt.figure(figsize = (12,12))
		for i in range(30):
			plt.plot(processing_rank_list_of_node(i, ini_rank_list), label = 'node {}'.format(i))
		plt.legend()
		plt.xlabel('Iterations', fontsize = 15)
		plt.ylabel('Quality and Status Score', fontsize = 15)
		plt.title('Rank Changes Over Time', fontsize = 20)
		plt.show()

	def initial_rank_based_on_quality(self, graph):
		"""
		# Returns the initial rank for each node w.r.t their quality.
		# The result will be used to calculate the spearman correlation.
		"""
		quality_dict = {}
		for i in range(30):
			quality_dict[i] = graph.node[i]['quality']
		# sorted_quality: The result of the rank for all nodes at initial stage
		sorted_quality = OrderedDict(sorted(quality_dict.items(), key=lambda x: x[1]))
		sorted_quality = list(sorted_quality)

		# Convert sorted_quality to initial rank for each node w.r.t their quality
		quality_rank_list = []
		for i in range(30):
			quality_rank_list.append(sorted_quality.index(i))

		return quality_rank_list

	def last_round_rank(self, ulti_rank_list):
		"""
		Returns the rank of nodes at the final status, ordered by their status score.
		"""
		last_round_rank = []
		for i in range(30):
			last_round_rank.append(ulti_rank_list[-1].index(i))
		return last_round_rank