from collections import OrderedDict
import numpy as np
import pdb

from formula import external_attachment, error_func, status_dissimilarity, z_score_of_node, sd_thres, nominal_value, get_rank_of_round

def update_node_normal(to_node, graph, pre_graph, sd_max, graph_type, nominal_num, err_level,
					   self_fulfilling, w, s, phi, h, si, heuristic):
	"""
	Update the attachment and status of input node in all other types.
	"""
	
	q = graph.node[to_node]['quality']

	e = error_func()

	status = 0

	# Following the principle of heuristic: each actor with have different h according to their rank
	if heuristic:
		for from_node in graph.neighbors(to_node):
			if sd_thres(graph, sd_max, from_node, to_node, h):
				attachment_total = external_attachment(from_node, to_node, pre_graph)
				if self_fulfilling:
					perceived_quality = (1 - w) * ((1-phi) * q + phi * pre_graph.node[to_node]['z_score']) + w * attachment_total
				elif graph_type == 'random' or graph_type == 'nominal':
					perceived_quality = (1 - w) * q + w * attachment_total
				elif graph_type == 'dyadic_error':
					perceived_quality = (1 - w) * (q + error_func(err_level)) + w * attachment_total
				elif graph_type == 'collective_error':
					perceived_quality = (1 - w) * (q + e) + w * attachment_total

				attachment_new = (perceived_quality + s * pre_graph[to_node][from_node]['weight']) / (2 * s)
				status += attachment_new
				graph[from_node][to_node]['weight'] = attachment_new

			else:
				attachment_new = 0
				status += 0
				graph[from_node][to_node]['weight'] = attachment_new

	# Normal update, without considering h
	elif sd_max == None:
		for from_node in graph.neighbors(to_node):
			attachment_total = external_attachment(from_node, to_node, pre_graph)
			if graph_type != 'ddd':
				if self_fulfilling:
					perceived_quality = (1 - w) * ((1-phi) * q + phi * pre_graph.node[to_node]['z_score']) + w * attachment_total
				elif graph_type == 'random' or graph_type == 'nominal':
					perceived_quality = (1 - w) * q + w * attachment_total
				elif graph_type == 'dyadic_error':
					perceived_quality = (1 - w) * (q + error_func(err_level)) + w * attachment_total
				elif graph_type == 'collective_error':
					perceived_quality = (1 - w) * (q + e) + w * attachment_total

				attachment_new = (perceived_quality + s * pre_graph[to_node][from_node]['weight']) / (2 * s)
				status += attachment_new
				graph[from_node][to_node]['weight'] = attachment_new
			else:
				ddd = pre_graph[from_node][to_node]['weight'] - pre_graph[to_node][from_node]['weight']
				perceived_quality = (1 - w) * (q + e) + w * attachment_total
				if ddd <= 0:
					attachment_new = perceived_quality
				else:
					attachment_new = perceived_quality - si * ddd
				status += attachment_new
				graph[from_node][to_node]['weight'] = attachment_new

	# Update with fixed h
	else:
		for from_node in graph.neighbors(to_node):
			if sd_thres(graph, sd_max, from_node, to_node, h):
				attachment_total = external_attachment(from_node, to_node, pre_graph)
				if self_fulfilling:
					perceived_quality = (1 - w) * ((1-phi) * q + phi * pre_graph.node[to_node]['z_score']) + w * attachment_total
				elif graph_type == 'random' or graph_type == 'nominal':
					perceived_quality = (1 - w) * q + w * attachment_total
				elif graph_type == 'dyadic_error':
					perceived_quality = (1 - w) * (q + error_func(err_level)) + w * attachment_total
				elif graph_type == 'collective_error':
					perceived_quality = (1 - w) * (q + e) + w * attachment_total

				attachment_new = (perceived_quality + s * pre_graph[to_node][from_node]['weight']) / (2 * s)
				status += attachment_new
				graph[from_node][to_node]['weight'] = attachment_new
			else:
				attachment_new = pre_graph[from_node][to_node]['weight']
				status += attachment_new
				graph[from_node][to_node]['weight'] = attachment_new

	graph.node[to_node]['status'] = status

def update_node_category(to_node, graph, pre_graph, sim_dict, w, s):
	"""	Update the attachment and status of input node in 'category' type.	"""

	q = graph.node[to_node]['quality']

	status = 0

	for from_node in graph.neighbors(to_node):
		c = sim_dict[(graph.node[from_node]['category'], graph.node[to_node]['category'])] / sim_dict[max(sim_dict.keys(), key=(lambda k: sim_dict[k]))]

		attachment_total = external_attachment(from_node, to_node, pre_graph)
		perceived_quality = (1 - w) * q * c + w * attachment_total

		attachment_new = (perceived_quality + s * pre_graph[to_node][from_node]['weight']) / (2 * s)

		status += attachment_new

		graph[from_node][to_node]['weight'] = attachment_new

	graph.node[to_node]['status'] = status

def update_graph_normal(graph, graph_type, nominal_num, err_level, self_fulfilling, w, s,
						phi, h, si, heuristic, update_node_normal):
	"""	Update all the actors in the graph (except 'category' graph) for this round.	"""
	
	pre_graph = graph.copy()

	if heuristic:
		sd_max = status_dissimilarity(pre_graph)
		different_h = np.arange(1/30, 1 + 1/30, step = 1/30).tolist()
		node_and_h = zip(rank_of_round, different_h)

		for to_node_and_h in node_and_h:
			update_node_normal( to_node_and_h[0], graph, pre_graph, sd_max, graph_type, nominal_num, err_level,
								self_fulfilling, w, s, phi, to_node_and_h[1], si, heuristic)
		for j in range(30):
			graph.node[j]['z_score'] = z_score_of_node(j, graph)

		return graph

	else:
		if (h > 0) & (h < 1):
			sd_max = status_dissimilarity(pre_graph)
		elif h == 1:
			sd_max = None
		for to_node in range(30):
			update_node_normal( to_node, graph, pre_graph, sd_max, graph_type, nominal_num, err_level,
								self_fulfilling, w, s, phi, h, si, heuristic)

		for j in range(30):
			graph.node[j]['z_score'] = z_score_of_node(j, graph)

		return graph

def update_graph_category(graph, sim_dict, w, s, update_node_category):
	"""	Update all the actors in the graph (only for 'category' graph) for this round.	"""
	pre_graph = graph.copy()

	for to_node in range(30):
		update_node_category(to_node, graph, pre_graph, sim_dict, w, s)

	for j in range(30):
		graph.node[j]['z_score'] = z_score_of_node(j, graph)

	return graph