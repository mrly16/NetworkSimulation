from graphs import graph_rd, graph_dyadic_err, graph_collective_err, graph_cat, graph_ddd, graph_nominal
from update_node import *

graph_specs = { 'random': { 'setup': graph_rd,
							'update_graph': update_graph_normal},

				'nominal': { 'setup': graph_nominal,
							 'update_graph': update_graph_normal},

				'dyadic_error': {'setup': graph_dyadic_err,
								'update_graph': update_graph_normal},

				'collective_error': {'setup': graph_collective_err,
									'update_graph': update_graph_normal},

				'category': {'setup': graph_cat,
							'update_graph': update_graph_category},

				'ddd': {'setup': graph_ddd,
						'update_graph': update_graph_normal},
			  }