import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import spearmanr
import pandas as pd
import tqdm
import itertools

from graph_specs import graph_specs
from update_node import update_node_category, update_node_normal
from formula import *
from utilities import *

""" 
HOW TO USE THE FILES:
--------------------------------------------------------------------------------------------------------------------
main.py: Instantiate the class NetworkSimulation with different initial parameters for constructing different graph, 
the class contains functions to calculate correlation, gini correlation, average converging time, status quality reordering
and status quality gap, and can be called to generate the graphs according to requirements.
--------------------------------------------------------------------------------------------------------------------
formula.py: Contains the functions for doing the small tasks in updating the graph, calculating necessary statistics, and
--------------------------------------------------------------------------------------------------------------------
update_node.py: Contains functions to update each nodes in each round and functions to update the graph.
--------------------------------------------------------------------------------------------------------------------
ultimate_simulation.py: For the 1210 * 100 times simulation.
--------------------------------------------------------------------------------------------------------------------
utilities.py: Mainly contains function to visualize the results or intermediate result.
"""


class NetworkSimulation(object):
	""" 
	NetworkSimulation contains a generic network simulation class that can create network graph
	of different type, upgrate and simulate the dynamic changes of network over time.
	"""
	def __init__(self, graph_type = 'random', 	# Options: 'random', 'dyadic_error', 'collective_error', 'category', 'ddd'

				 self_fulfilling = False,		# Whether the actor follows the self-fulfilling rule, detailed on self.phi

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

				 heuristic = False				# Whether the actors will follow heuristic principle
				 								# Lower status actors would be assigned higher h, vice versa
				 ):

		self.graph_type = graph_type
		self.self_fulfilling = self_fulfilling
		self.nominal_num = nominal_num
		self.err_level = err_level
		self.w = w
		self.s = s
		self.phi = phi
		self.h = h
		self.si = si
		self.category_num = category_num
		self.heuristic = heuristic
		self.graphs = []

		if self.graph_type not in graph_specs.keys():
			raise ValueError('Please specify one of the following graphs: ' + ','.join(graph_specs.keys()))

		self.update_graph = graph_specs[self.graph_type]['update_graph']

		ini_category = np.arange(self.category_num, step = 1).tolist()
		category = [x + 1 for x in ini_category]

		self.category = category
		self.sim_dicts = []

	def graph_setup(self):
		""" Set up the graph with initial parameters."""

		G = graph_specs[self.graph_type]['setup'](self.err_level, self.nominal_num, self.category_num)

		for i in range(30):
			G.node[i]['status'] = calculate_status(i, G)

		for j in range(30):
			G.node[j]['z_score'] = z_score_of_node(j, G)

		# quality_rank_list = self.initial_rank_based_on_quality(G)

		self.graphs.append(G.copy())

		return G

	def simulation(self):
		"""Iterate the update process multiple times to simulate the development of network."""
		local_graphs = []

		G = self.graph_setup()

		G = status_standardization(G)   #   w(T.T)w

		local_graphs.append(G)

		for tt in range(20):
			G = self.update_graph(G, self.graph_type, self.nominal_num, self.err_level, self.self_fulfilling, self.w, self.s, self.phi, self.h, self.si, self.heuristic, update_node_normal)
			G = status_standardization(G)     # w(T.T)w
			self.graphs.append(G.copy())
			# use local_graphs to keep track results of each simulation
			local_graphs.append(G.copy())

		return self.calculate_correlation(local_graphs), self.gini_correlation(), self.average_converging_time(), abs(self.status_quality_gap_list()[-1])

		# return self.graphs
	def simulation_category(self):
		"""Iterate the update process multiple times to simulate the development of network.(For category only)"""
		G = self.graph_setup()

		G = status_standardization(G)   #   w(T.T)w

		# Create a matrix to keep track of category choices of each node in every round
		choice_history = np.zeros(shape = (30, 20), dtype = np.int8)

		# Complete the choice_history matrix before doing the updating, then feed the choice to each node during simulation
		for counter in range(20):
			for k in range(30):
				c = np.random.choice(self.category)
				G.node[k]['category'] = c
				choice_history[k][counter] = c

		self.choice_history = choice_history

		for tt in range(20):
			# from the second round, we start to update the graph
			if tt > 0:
				# re-calculating the similarity of each pair of nodes at the begining of each round
				sim_dict = {}
				category_pair = [(i, j) for i in self.category for j in self.category]
				for each_pair in category_pair:
					if each_pair[0] != each_pair[1]:
						sim_dict[each_pair] = calculate_similarity(self.choice_history[:, :tt+1], each_pair[0], each_pair[1])
					else:
						sim_dict[each_pair] = 1

				# update the graph
				G = self.update_graph(G, sim_dict, self.w, self.s, update_node_category)

				# standardizing the status of all nodes in the graph
				G = status_standardization(G)

				# assgined each node with a new category
				for i in range(30):
					G.node[i]['category'] = choice_history[i][tt]
				self.graphs.append(G.copy())
				self.sim_dicts.append(sim_dict)
			# for the first round we do nothing
			else:
				for i in range(30):
					G.node[i]['category'] = choice_history[i][tt]
				self.graphs.append(G.copy())


	def calculate_correlation(self, graph_list):
		""" Calculate the correlation of the class"""

		#
		rank_start = initial_rank_based_on_quality(graph_list[0])
		rank_last = get_rank_of_round(graph_list[-1])
		return spearmanr(rank_start, rank_last)[0]

	def multiple_simulations(self):
		"""
		This function can be called directly for simulation with different parameters.
		Simulating the network development for multiple times and get the averaged result.
		"""
		cor_list = []
		gini_list = []
		avg_conv_list = []
		abs_sqg_list = []
		for ii in tqdm.trange(100):
			if self.graph_type != 'category':
				cor, gini, avg_conv, abs_sqg = self.simulation()
				cor_list.append(cor)
				gini_list.append(gini)
				avg_conv_list.append(avg_conv)
				abs_sqg_list.append(abs_sqg)
			else:
				self.simulation_category()
		return np.mean(cor_list), np.mean(gini_list), np.mean(avg_conv_list), np.mean(abs_sqg_list)

	def status_quality_reordering_list(self):
		return [status_quality_reordering(each_graph) for each_graph in self.graphs]

	def status_quality_gap_list(self):
		return [status_quality_gap(each_graph) for each_graph in self.graphs]

	def gini_correlation(self):
		return gini(self.graphs[-1])

	def average_converging_time(self):
		status_rank_list = [get_rank_of_round(each_graph) for each_graph in self.graphs]
		converge = 0
		for idx, each_list in enumerate(status_rank_list):
			if idx + 1 <= 20 - 1:
				if each_list == status_rank_list[idx + 1]:
					converge += 1
					break
				else:
					converge += 1
		return converge

	def output_matrix_for_category(self):
		output_matrix = []
		tt= 0
		for each_graph in self.graphs[1:]:
			output_list = []
			if tt > 0:
				for counter in range(30):
					output_list.append({	'round': tt,
											'perceived quality': get_node_perceived_quality(each_graph, counter, self.sim_dicts[tt-1], self.w),
											'status score': each_graph.node[counter]['status'],
											'status change': each_graph.node[counter]['status'] - self.graphs[tt-1].node[counter]['status'],
											'rank': get_status_rank_of_node(each_graph, counter),
											'current category': self.choice_history[counter, tt],
											'choice history': self.choice_history[counter, :tt+1].tolist(),
											'average similarity': avg_category_similarity(self.choice_history[:counter+1, :tt+1][0], self.sim_dicts[tt-1])})
			else:
				for counter in range(30):
					output_list.append({    'round': tt,
											'perceived quality': get_node_perceived_quality(each_graph, counter, self.sim_dicts[tt-1], self.w),
											'status score': each_graph.node[counter]['status'],
											'status change': 0,
											'rank': get_status_rank_of_node(each_graph, counter),
											'current category': self.choice_history[counter, tt],
											'choice history': self.choice_history[counter, :tt+1].tolist(),
											'average similarity': 0})
			tt += 1
			output_matrix.append(output_list)
		return output_matrix