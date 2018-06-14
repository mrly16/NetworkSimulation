import numpy as np

''' This file contains general formulas for calculation '''
''' _author_: liangyi '''

# Function to return all possible node pairs
def all_pairs(actor_num = 30):
	pair_list = []
	for i in range(actor_num):
		for j in range(actor_num):
			if i != j:
				pair_list.append((i,j))
	# pair_list contains list of tuple. e.g (0, 1)
	return pair_list

# Return a random value x ~ N(0, 1)
def normal_distribution():
	return np.random.normal(0,1)

# Return a random value x ~ N(0, 1)
# According to the paper, this function should return a random value a~U(0, 1)
# (Alternative) return np.random.normal(-err_level, err_level)
def error_func(err_level = 1):
	return np.random.normal(0, err_level)

# Return a random value from given list of values.
def nominal_value(nominal_num = 2):
    if nominal_num == 2:
        return np.random.choice([3, -3])
    if nominal_num == 3:
        return np.random.choice([3, 0, -3])
    if nominal_num == 4:
        return np.random.choice([3, 1, -1, -3])

# Calculate status score for the node
# Summation of attachment to receiver(node j)
def calculate_status(to_node, graph):
	status_score = 0
	for each_node in graph.neighbors(to_node):
		status_score += graph[each_node][to_node]['weight']
	return status_score

# Return the similarity of node i and node j
# Similarity = count(node i CAP node j) / (count(node i) + count(node j))
def calculate_similarity(matrix, i, j):
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

# Self-fulfilling: individuals will adjust their behavior in correspondence with how they are treated
# z-score: standardizd value, (Xi - x.mean()) / x.std()
def z_score_of_node(node, graph):
	score = []
	for i in range(30):
		score.append(graph.node[i]['status'])
	return (graph.node[node]['status'] - np.mean(score)) / np.std(score)

# Return the average attachment of a node(all attachment to target node except)
# Summation of attachment from each node(node k) to receiver(node j) where k != i, j
def external_attachment(from_node, to_node, graph):
	weighted_average = 0
	for each_node in graph.neighbors(to_node):
		if each_node != from_node:
			weighted_average += graph[each_node][to_node]['weight']
	return weighted_average

# Find the biggest difference(absolute value) among status score of the nodes
# SD: status dissimilarity
def status_dissimilarity(graph, h):
	status_list = []
	for i in range(30):
		status_list.append(graph.node[i]['status'])
	return (max(status_list) - min(status_list)) * h

# Return true if status dissimilarity of two nodes are smaller than threshold of SD
def sd_thres(graph, sd_max, from_node, to_node):
	diff = abs(graph.node[from_node]['status'] - graph.node[to_node]['status'])
	if diff <= sd_max:
		return True
	else:
		return False


