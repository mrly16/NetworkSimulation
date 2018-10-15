import networkx as nx
import numpy as np

from formula import all_pairs, normal_distribution, error_func, nominal_value

""" This file contains method to construct all types of graph
	Graph type: {
		'graph_rd': attachments are all randomized,
		'graph_dyadic_err': errors are NOT shared across actors,
		'graph_collective_err': errors are shared across acotors,
		'graph_cat': all actors choose a different category to invest at different time stage,
		'graph_ddd': errors are NOT shared across actors, and actors are vengeful
	}
"""
""" _author_: liangyi """

def graph_rd(err_level, nominal_num, category_num):
	"""
	Graph for randomized network
	The attachment between nodes are random value from N~(0, 1)
	At initialization, the quality and attachment are all independent randomized value
	"""

	G = nx.DiGraph()
	for i in range(30):
		q = normal_distribution()
		G.add_node(i, quality = q)
	# pdb.set_trace()
	pair_list = all_pairs(30)
	for pair in pair_list:
		# initialize the weight as random value that follows a normal distribution
		attachment = normal_distribution()
		G.add_edge(pair[0], pair[1], weight = attachment)
	return G

# construct a directed network with all nodes connected
def graph_nominal(err_level, nominal_num, category_num):
    G = nx.DiGraph()
    for i in range(30):
        q = normal_distribution()
        n = nominal_value(nominal_num)
        G.add_node(i, quality = q, nominal = n)
    pair_list = all_pairs(30)
    for pair in pair_list:
        attachment = G.node[pair[1]]['nominal']
        G.add_edge(pair[0], pair[1], weight = attachment)
    return G

def graph_dyadic_err(err_level, nominal_num, category_num):
	"""
	Graph for dyadic level error network
	The attachment between nodes are the sum of quality of the to_node and an error
	Error of the attachment is err~N(0, x), x is the error level
	At initialization, the quality and attachment are all independent randomized value
	"""

	G = nx.DiGraph()
	for i in range(30):
		q = normal_distribution()
		G.add_node(i, quality = q)
	pair_list = all_pairs(30)
	for pair in pair_list:
		# initialize the weight as the sum of quality of the to_node and an error 
		attachment = G.node[pair[1]]['quality'] + error_func(err_level)
		G.add_edge(pair[0], pair[1], weight = attachment)
	return G

def graph_collective_err(err_level, nominal_num, category_num):
	"""
	# Graph for collective level error network
	# Error of the attachment is err~N(0, x), x is the error level
	# At initialization, the quality and attachment are all independent randomized value
	"""

	G = nx.DiGraph()
	for i in range(30):
		q = normal_distribution()
		e = error_func(err_level)
		G.add_node(i, quality = q, error = e)
	pair_list = all_pairs(30)
	for pair in pair_list:
		# initialize the weight as the sum of quality of the to_node and the error of the to_node
		attachment = G.node[pair[1]]['quality'] + G.node[pair[1]]['error']
		G.add_edge(pair[0], pair[1], weight = attachment)
	return G

def graph_cat(err_level, nominal_num, category_num):
	G = nx.DiGraph()
	for i in range(30):
		# c stands for category choice
		q = normal_distribution()
		G.add_node(i, quality = q)
	pair_list = all_pairs(30)
	for pair in pair_list:
		G.add_edge(pair[0], pair[1], weight = G.node[pair[1]]['quality'])
	return G

def graph_ddd(err_level, nominal_num, category_num):
	G = nx.DiGraph()
	for i in range(30):
		q = normal_distribution()
		G.add_node(i, quality = q)
	pair_list = all_pairs(30)
	for pair in pair_list:
		# initialize the weight as the sum of quality of the to_node and the error of the to_node
		attachment = G.node[pair[1]]['quality']
		G.add_edge(pair[0], pair[1], weight = attachment)
	return G