import numpy as np
from collections import OrderedDict

def all_pairs(actor_num = 30):
	""" Function to return all possible node pairs"""

	pair_list = []
	for i in range(actor_num):
		for j in range(actor_num):
			if i != j:
				pair_list.append((i,j))
	# pair_list contains list of tuple. e.g (0, 1)
	return pair_list

def normal_distribution():
	""" Return a random value x ~ N(0, 1)"""

	return np.random.normal(0,1)

# According to the paper, this function should return a random value a~U(0, 1)
# (Alternative) return np.random.normal(-err_level, err_level)
def error_func(err_level = 1):
	""" Return a random value x ~ N(0, 1)"""

	return np.random.normal(0, err_level)

def nominal_value(nominal_num = 2):
	""" Return a random value from given list of values."""

	if nominal_num == 2:
		return np.random.choice([3, -3])
	if nominal_num == 3:
		return np.random.choice([3, 0, -3])
	if nominal_num == 4:
		return np.random.choice([3, 1, -1, -3])

# Summation of attachment to receiver(node j)
def calculate_status(to_node, graph):
	""" Calculate status score for the node"""

	status_score = 0
	for each_node in graph.neighbors(to_node):
		status_score += graph[each_node][to_node]['weight']
	return status_score

# Similarity = count(node i CAP node j) / (count(node i) + count(node j))
def calculate_similarity(matrix, i, j):
	""" Return the similarity of node i and node j"""

	matrix = matrix.tolist()
	count_i = 0
	count_j = 0
	ij_cap = 0
	for k in range(len(matrix)):
		count_i += matrix[k].count(i)
		count_j += matrix[k].count(j)
		if i in matrix[k] and j in matrix[k]:
			ij_cap += 1
	if (count_i + count_j) != 0:
		similarity = ij_cap / (count_i + count_j)
	else:
		similarity = 0
	return similarity

# z-score: standardizd value, (Xi - x.mean()) / x.std()
def z_score_of_node(node, graph):
	""" individuals will adjust their behavior in correspondence with how they are treated"""

	score = []
	for i in range(30):
		score.append(graph.node[i]['status'])
	return (graph.node[node]['status'] - np.mean(score)) / np.std(score)

# Summation of attachment from each node(node k) to receiver(node j) where k != i, j
def external_attachment(from_node, to_node, graph):
	"""Return the average attachment of a node(all attachment to target node except)"""

	weighted_average = 0
	for each_node in graph.neighbors(to_node):
		if each_node != from_node:
			weighted_average += graph[each_node][to_node]['weight']
	return weighted_average

# SD: status dissimilarity
def status_dissimilarity(graph):
	""" Find the biggest difference(absolute value) among status score of the nodes"""

	status_list = []
	for i in range(30):
		status_list.append(graph.node[i]['status'])
	return (max(status_list) - min(status_list))

def sd_thres(graph, sd_max, from_node, to_node, h):
	""" Decide whether there should be an attachment between the from_node and to_node based on their status_dissimilarity"""
	diff = abs(graph.node[from_node]['status'] - graph.node[to_node]['status'])
	# Return true if status dissimilarity of two nodes are smaller than threshold of SD
	if diff <= sd_max * h:
		return True
	else:
		return False

def get_rank_of_round(graph):
	""" Return the rank of all nodes in a graph as a list"""
	rank_dict = {}
	for i in range(30):
		rank_dict[i] = graph.node[i]['status']
	sorted_rank =  list(OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1])).keys())

	status_rank_list = [sorted_rank.index(i) for i in range(30)]

	return status_rank_list

def status_standardization(graph):
	""" Return the graph with normalized status of each node in the graph"""
	status_list = [graph.node[i]['status'] for i in range(30)]
	mean = np.mean(status_list)
	std = np.std(status_list)
	for j in range(30):
		graph.node[j]['status'] = (graph.node[j]['status'] - mean) / std
	return graph


def pairwise(iterable):
	""" Create a list of tuples from the input iterable. Each tuple consist of an element in the iterable and 
	its next element
	"""
	it_list = []
	it = iter(iterable)
	a = next(it, None)

	for b in it:
		it_list.append((a, b))
		a = b
		
	return it_list

def avg_category_similarity(history_list, sim_dict):
	l = pairwise(history_list)
	return sum([sim_dict[pair] for pair in l]) / len(l)

def gini(graph):
	"""Calculate the Gini coefficient of a numpy array."""
	# from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

	status_list = [graph.node[i]['status'] for i in range (30)]
	
	array = np.array(status_list, dtype=np.float64)
	array = array.flatten() #all values are treated equally, arrays must be 1d
	if np.amin(array) < 0:
		array -= np.amin(array) #values cannot be negative
	array += 0.0000001 #values cannot be 0
	array = np.sort(array) #values must be sorted
	index = np.arange(1,array.shape[0]+1) #index per array element
	n = array.shape[0]#number of array elements
	return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def status_quality_reordering(graph):
	""" Return the status quality reordering of a certain graph"""
	return (1 - sum([(graph.node[i]['quality'] - graph.node[i]['status'])**2 for i in range(30)]) * 6 / (30*30*30 - 30))

def status_quality_gap(graph):
	""" Returns the status quality gap for a certain graph"""
	return (sum([abs(graph.node[i]['status'] - graph.node[i]['quality']) for i in range(30)]) / 30)

def get_node_perceived_quality(graph, to_node, sim_dict, w):
	""" Get the perceived quality of a node in a category graph"""

	pq_list = []
	q = graph.node[to_node]['quality']
	for from_node in graph.neighbors(to_node):
		# c = similarity of node i and node j divided by the largest similarity
		c = sim_dict[(graph.node[from_node]['category'], graph.node[to_node]['category'])] / sim_dict[max(sim_dict.keys(), key=(lambda k: sim_dict[k]))]

		# Calculate perceived quality
		attachment_total = external_attachment(from_node, to_node, graph)
		perceived_quality = (1 - w) * q * c + w * attachment_total

		pq_list.append(perceived_quality)

	return np.mean(pq_list)
		
def get_status_rank_of_node(graph, node):
	""" Get the rank of certain node for a certain graph"""

	rank_dict = {}
	for i in range(30):
		rank_dict[i] = graph.node[i]['status']
	# maybe should be reversed w(T.T)w
	sorted_rank =  list(OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1])).keys())

	return sorted_rank.index(node)