# k-Means clustering for Normal Distributions - Almost from scratch!

import numpy as np
import scipy as sp


def distanceFunction1D(X,Y):
	dist = np.zeros((X.shape[0], Y.shape[0]))
	#print X.shape, Y.shape
	for i in range(X.shape[0]):
		for j in range(Y.shape[0]):
			#print w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1])
			dist[i,j] = (w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1]))

	return dist

