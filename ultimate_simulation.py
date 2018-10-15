import numpy as np
import itertools
from main import NetworkSimulation
def ultimate_iteration():
	h = np.arange(0.1, 1.1, step = 0.1).tolist()
	w = np.arange(0.1, 1.1, step = 0.1).tolist()
	s = np.arange(0.1, 1.1, step = 0.1).tolist()
	combinitions = itertools.product(h, w, s)

	cor_list = []
	gini_list = []
	avg_conv_list = []
	abs_sqg_list = []
	for each in combinitions:
		params = {'h':each[0], 'w':each[1], 's':each[2]}
		g = NetworkSimulation(**params)
		cor, gini, avg_conv, abs_sqg = g.multiple_simulations()
		cor_list.append(cor)
		gini_list.append(gini)
		avg_conv_list.append(avg_conv)
		abs_sqg_list.append(abs_sqg)
	return cor_list, gini_list, avg_conv_list, abs_sqg_list