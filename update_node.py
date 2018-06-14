from collections import OrderedDict
import pdb

from formula import external_attachment, error_func, status_dissimilarity, z_score_of_node, sd_thres, nominal_value

def update_node_normal(to_node, graph, pre_graph, sd_max, graph_type, nominal_num, err_level, self_fulfilling, w, s, phi, si):
	"""
	Update the attachment and status of input node in all other types.
	"""
	
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
				elif graph_type == 'random' or graph_type == 'nominal':
					perceived_quality = (1 - w) * q + w * attachment_total
				elif graph_type == 'dyadic_error':
					perceived_quality = (1 - w) * (q + error_func(err_level)) + w * attachment_total
				elif graph_type == 'collective_error':
					perceived_quality = (1 - w) * (q + e) + w * attachment_total
				# elif graph_type == 'nominal':
				# 	perceived_quality = (1 - w) * (q + nominal_value(nominal_num)) + w * attachment_total

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
				elif graph_type == 'random' or graph_type == 'nominal':
					perceived_quality = (1 - w) * q + w * attachment_total
				elif graph_type == 'dyadic_error':
					perceived_quality = (1 - w) * (q + error_func(err_level)) + w * attachment_total
				elif graph_type == 'collective_error':
					perceived_quality = (1 - w) * (q + e) + w * attachment_total
				# elif graph_type == 'nominal':
				# 	perceived_quality = (1 - w) * (q + nominal_value(nominal_num)) + w * attachment_total

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

def update_node_category(to_node, graph, pre_graph, sim_dict, w, s):
	"""
	Update the attachment and status of input node in 'category' type.
	"""

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

def update_graph_normal(graph, graph_type, nominal_num, err_level, self_fulfilling, w, s, phi, h, si, update_node_normal):
	"""
	Update all the actors in the graph (except 'category' graph) for this round.
	"""
	
	pre_graph = graph.copy()
	rank_dict = {}
	if (h > 0) & (h < 1):
		sd_max = status_dissimilarity(pre_graph, h)
	elif h == 1:
		sd_max = None
	for to_node in range(30):
		update_node_normal(to_node, graph, pre_graph, sd_max, graph_type, nominal_num, err_level, self_fulfilling, w, s, phi, si)
		rank_dict[to_node] = graph.node[to_node]['status']
	sorted_rank = OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1]))

	# Update the z_score of each node at the end of update stage
	for j in range(30):
		graph.node[j]['z_score'] = z_score_of_node(j, graph)

	# print('the rank of this round is', list(sorted_rank.keys()))
	return graph, list(sorted_rank.keys())

def update_graph_category(graph, sim_dict, w, s, update_node_category):
	"""
	Update all the actors in the graph (only for 'category' graph) for this round.
	"""
	pre_graph = graph.copy()
	rank_dict = {}
	for to_node in range(30):
		update_node_category(to_node, graph, pre_graph, sim_dict, w, s)
		rank_dict[to_node] = graph.node[to_node]['status']
	sorted_rank = OrderedDict(sorted(rank_dict.items(), key=lambda x: x[1]))

	# Update the z_score of each node at the end of update stage
	for j in range(30):
		graph.node[j]['z_score'] = z_score_of_node(j, graph)

	# print('the rank of this round is', list(sorted_rank.keys()))
	return graph, list(sorted_rank.keys())
