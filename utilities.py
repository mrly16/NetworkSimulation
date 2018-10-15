import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import numpy as np

# Process rank of each round
# Find the list of rank status of each node
# Desired output format: node 0: [2, 1, 2, 3, 2, 2, 2, ... 2]
# Number in the list denotes the rank of the node in each round
def processing_rank_list_of_node(node, ini_rank_list):
	node_rank_list = []
	# rank_list: The result of the rank for all nodes in each round, ordered from the lowest to the highest
	# This function returns the rank of each round for each node
	for rank in ini_rank_list:
		node_rank_list.append(rank.index(node)+1)
	return node_rank_list

# Plot the changes of rank for each node over time
def rank_change_over_time(ini_rank_list):
	plt.figure(figsize = (12,12))
	for i in range(30):
		plt.plot(processing_rank_list_of_node(i, ini_rank_list), label = 'node {}'.format(i))
	plt.legend()
	plt.xlabel('Iterations', fontsize = 15)
	plt.ylabel('Quality and Status Score', fontsize = 15)
	plt.title('Rank Changes Over Time', fontsize = 20)
	plt.show()

# Returns the initial rank for each node w.r.t their quality
# Used to calculate the spearman correlation
def initial_rank_based_on_quality(graph):
	# sorted_quality: The result of the rank for all nodes at initial stage
	quality_dict = {}
	for i in range(30):
		quality_dict[i] = graph.node[i]['quality']
	sorted_quality = OrderedDict(sorted(quality_dict.items(), key=lambda x: x[1]))
	sorted_quality = list(sorted_quality)

	# Convert sorted_quality to initial rank for each node w.r.t their quality
	quality_rank_list = []
	for i in range(30):
		quality_rank_list.append(sorted_quality.index(i))

	return quality_rank_list

# Rturns the rank of nodes at the final status, ordered by their status score
def last_round_rank(ulti_rank_list):
	last_round_rank = []
	for i in range(30):
		last_round_rank.append(ulti_rank_list[-1].index(i))
	return last_round_rank

def plot_three_sq_graph(lst):
	plt.figure(figsize = (20, 12))
	plt.plot([x for x in range(len(lst))], lst)
	plt.xticks(np.arange(0, len(lst)))
	plt.show()

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

def last_round_rank(self, ulti_rank_list):
	"""
	Returns the rank of nodes at the final status, ordered by their status score.
	"""
	last_round_rank = []
	for i in range(30):
		last_round_rank.append(ulti_rank_list[-1].index(i))
	return last_round_rank