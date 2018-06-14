from collections import OrderedDict
import numpy as np
from scipy.stats import spearmanr

''' This file contains the main method for simulation '''
''' _author_: liangyi '''

# Function to update input node, 
# Called in every iteration process and update the attachment of each actor
# Default w(omega) = 0.20, s = 1
# grahp: network graph of current round, pre_graph, network graph of previous round.
def update_node(to_node,
				graph,
				pre_graph,
				sd_max,
				graph_type = 'random',
				err_level = 1,
				self_fulfilling = False,
				w =0.2,
				s = 1,
				phi = 0.5,
				si = 0.5):
	q = graph.node[to_node]['quality']

	if (graph_type == 'collective_error') | (graph_type == 'ddd'):
		e = error_func()

	status = 0
	if sd_max == None:
		for from_node in graph.neighbors(to_node):
			# Calculate perceived quality
			attachment_total = external_attachment(from_node, to_node, pre_graph)
			if graph_type != 'ddd':
				if self_fulfilling:
					perceived_quality = (1 - w) * ((1-phi) * q + phi * pre_graph.node[to_node]['z_score']) + w * attachment_total
				elif graph_type == 'random':
					perceived_quality = (1 - w) * q + w * attachment_total
				elif graph_type == 'dyadic_error':
					perceived_quality = (1 - w) * (q + error_func(err_level)) + w * attachment_total
				elif graph_type == 'collective_error':
					perceived_quality = (1 - w) * (q + e) + w * attachment_total

				# Generate best-response attachments
				# pre_graph[to_node][from_node]['weight']: attachment from receiver(node j) to sender(node i)
				attachment_new = (perceived_quality + s * pre_graph[to_node][from_node]['weight']) / (2 * s)

				# Also keep track of the status
				status += attachment_new

				# Update the attachment between sender(node i) and receiver(node j) of current graph
				graph[from_node][to_node]['weight'] = attachment_new
			else:
				ddd = pre_graph[from_node][to_node]['weight'] - pre_graph[to_node][from_node]['weight']
				perceived_quality = (1 - w) * (q + e) + w * attachment_total
				if ddd <= 0:
					attachment_new = perceived_quality
				else:
					attachment_new = perceived_quality - si * ddd
				# Also keep track of the status
				status += attachment_new

				# Update the attachment between sender(node i) and receiver(node j) of current graph
				graph[from_node][to_node]['weight'] = attachment_new

	else:
		for from_node in graph.neighbors(to_node):
			if sd_thres(graph, sd_max, from_node, to_node):
				# Calculate perceived quality
				attachment_total = external_attachment(from_node, to_node, pre_graph)
				if self_fulfilling:
					perceived_quality = (1 - w) * ((1-phi) * q + phi * pre_graph.node[to_node]['z_score']) + w * attachment_total
				elif graph_type == 'random':
					perceived_quality = (1 - w) * q + w * attachment_total
				elif graph_type == 'dyadic_error':
					perceived_quality = (1 - w) * (q + error_func(err_level)) + w * attachment_total
				elif graph_type == 'collective_error':
					perceived_quality = (1 - w) * (q + e) + w * attachment_total

				# Generate best-response attachments
				# pre_graph[to_node][from_node]['weight']: attachment from receiver(node j) to sender(node i)
				attachment_new = (perceived_quality + s * pre_graph[to_node][from_node]['weight']) / (2 * s)

				# Also keep track of the status
				status += attachment_new

				# Update the attachment between sender(node i) and receiver(node j) of current graph
				graph[from_node][to_node]['weight'] = attachment_new

			else:
				attachment_new = pre_graph[from_node][to_node]['weight']

				# Also keep track of the status
				status += attachment_new

				# Update the attachment between sender(node i) and receiver(node j) of current graph
				graph[from_node][to_node]['weight'] = attachment_new

	# Update the status score of the target node (node j)
	graph.node[to_node]['status'] = status

def update_node_cat(to_node,
				graph,
				pre_graph,
				sim_dict,
				w =0.2,
				s = 1):
	q = graph.node[to_node]['quality']

	status = 0
	for from_node in graph.neighbors(to_node):
		# c = similarity of node i and node j divided by the largest similarity
		# (TODO) adjust the formula for getting the c so that it wouldn't be too small
		c = sim_dict[(from_node, to_node)] / sim_dict[max(sim_dict.keys(), key=(lambda k: sim_dict[k]))]

		# Calculate perceived quality
		attachment_total = external_attachment(from_node, to_node, pre_graph)
		perceived_quality = (1 - w) * q * c + w * attachment_total

		# Generate best-response attachments
		# pre_graph[to_node][from_node]['weight']: attachment from receiver(node j) to sender(node i)
		attachment_new = (perceived_quality + s * pre_graph[to_node][from_node]['weight']) / (2 * s)

		# Also keep track of the status
		status += attachment_new

		# Update the attachment between sender(node i) and receiver(node j) of current graph
		graph[from_node][to_node]['weight'] = attachment_new
	# Update the status score of the target node (node j)
	graph.node[to_node]['status'] = status


# Process to finish the update for all nodes in the network for one round
def update_graph(graph, graph_type = 'random', err_level = 1, self_fulfilling = False, w = 0.2, s = 1, phi = 0.5, h = 1, si = 0.5):
	pre_graph = graph.copy()
	rank_dict = {}
	if (h > 0) & (h < 1):
		sd_max = status_dissimilarity(pre_graph, h)
	elif h == 1:
		sd_max = None
	for to_node in range(30):
		update_node(to_node, graph, pre_graph, sd_max, graph_type, err_level, self_fulfilling, w, s, phi, si)
		rank_dict[to_node] = graph.node[to_node]['status']
	sorted_rank = OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1]))

	# Update the z_score of each node at the end of update stage
	for j in range(30):
		graph.node[j]['z_score'] = z_score_of_node(j, graph)

	# print('the rank of this round is', list(sorted_rank.keys()))
	return graph, list(sorted_rank.keys())

# Process to finish the update for all nodes in the network for one round
def update_cat_graph(graph, sim_dict, w = 0.2, s = 1):
	pre_graph = graph.copy()
	rank_dict = {}
	for to_node in range(30):
		update_node_cat(to_node, graph, pre_graph, sim_dict, w, s)
		rank_dict[to_node] = graph.node[to_node]['status']
	sorted_rank = OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1]))

	# Update the z_score of each node at the end of update stage
	for j in range(30):
		graph.node[j]['z_score'] = z_score_of_node(j, graph)

	# print('the rank of this round is', list(sorted_rank.keys()))
	return graph, list(sorted_rank.keys())

# Set up the initial graph of different type, different w(social influence), different s(symmetry)
# Simulating the development of network graph with given initial status for given number of rounds
# By default it just returns the spearman correlation between initial rank and ultimate rank
def set_up_and_iterating(graph_type = 'random',
						actor_num = 30,
						iterating_times=20,
						err_level = 1,
						self_fulfilling = False,
						w = 0.2,
						s = 1,
						phi = 0.5,
						h = 1,
						si = 0.5):
	if graph_type == 'random':
		# Setting up the random graph
		G = graph_rd(actor_num)
	elif graph_type == 'dyadic_error':
		# Setting up the dyadic error graph
		G = graph_dyadic_err(actor_num, err_level)
	elif graph_type == 'collective_error':
		# Setting up the collective error graph		
		G = graph_collective_err(actor_num, err_level)
	elif graph_type == 'category':
		# Setting up the category graph	
		G, category = graph_cat(actor_num)
		# Create a matrix to keep track of the choice history of each actor
		choice_history = np.zeros(shape = (30, 20), dtype = np.int8)
	elif graph_type == 'ddd':
		# Setting up the DDD graph
		G = graph_collective_err(actor_num, err_level)


	# Calculate the status score at initialization stage
	for i in range(actor_num):
		G.node[i]['status'] = calculate_status(i, G)

	# At initialization stage, also calculate the standardized quality of each node by standardizing the status score		
	# pdb.set_trace()
	for j in range(actor_num):
		G.node[j]['z_score'] = z_score_of_node(j, G)

	quality_rank_list = initial_rank_based_on_quality(G)

	# Iterating for 20 rounds
	status_rank_list = []
	for tt in range(iterating_times):
		# For 
		if graph_type == 'category':
			for k in range(actor_num):
				# Update the choice history of tt round
				choice_history[k][tt] = G.node[k]['category']
				# At the beginning of each iteration, update the category of each node in advance
				G.node[k]['category'] = np.random.choice(category)

		if graph_type == 'category':
			if tt > 0:
				# To get the biggest similarity among all, a dictionary is created to record the similarity for all possible connection
				sim_dict = {}
				pair_list = all_pairs(actor_num)
				for each_pair in pair_list:
					sim_dict[each_pair] = calculate_similarity(choice_history, each_pair[0], each_pair[1])
				G, status_rank = update_cat_graph(G, sim_dict, w, s)
				# ipdb.set_trace()
			else:
				# Do nothing at the first round
				status_rank = quality_rank_list
		else:
			G, status_rank = update_graph(G, graph_type, err_level, self_fulfilling, w, s, phi, h, si)
		status_rank_list.append(status_rank)
	last_round_rank_list = last_round_rank(status_rank_list)

	# pdb.set_trace()
	correlation = spearmanr(quality_rank_list, last_round_rank_list)[0]
	return correlation

# This function can be called directly for simulation with different parameters
# {
	# 'simulation_times': times of dynamic network simulation with same parameters, 
	# 'graph_type': the type of the network simulation(random), 
	# 'actor_num': number of actors in the network,
	# 'iteration_times': times of network development rounds,
	# 'w': social influence,
	# 's': symmetry
# }
# Simulating the network development for multiple times and get the average correlation
def simulation( simulation_times = 30,
				graph_type = 'random',
				self_fulfilling = False,
				actor_num = 30,
				iteration_times = 20,
				err_level = 1,
				w = 0.2,
				s = 1,
				phi = 0.5,
				h = 1,
				si = 0.5):

	cor_list = []
	for ii in range(simulation_times):
		cor = set_up_and_iterating(graph_type, actor_num, iteration_times, err_level, self_fulfilling, w, s, phi, h, si)
		cor_list.append(cor)
	# pdb.set_trace()
	return np.mean(cor_list)
