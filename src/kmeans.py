# k-Means clustering for Normal Distributions - Almost from scratch!

import numpy as np
import scipy as sp
from ssUtils import *
from ImageOperations import *
import random

def distanceFunction1D(X,Y):
	"""
	Helper function to find distance between the 1d normal distributions.
	Used for kmeans on 1d distributions.
	"""
	dist = np.zeros((X.shape[0], Y.shape[0]))
	for i in range(X.shape[0]):
		for j in range(Y.shape[0]):
			dist[i,j] = (w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1]))

	return dist

def distanceFunction2D(X,Y):
	"""
	Helper function to find the distance between 2d normal distributions.
	This helper is used for the kmeans clustering of the 2d normal distributions
	"""
	dist = np.zeros((X.shape[0], Y.shape[0]))
	for i in range(X.shape[0]):
		for j in range(Y.shape[0]):
			dist[i,j] = (w2distance2D(X[i,0], X[i,1], Y[j,0], Y[j,1]))
	return dist


def testDistance1D():
	"""
	Test function
	"""
	mu1 = 50
	mu2 = 51
	mu3 = 240
 
	sigma1 = 1000
	sigma2 = 1050
	sigma3 = 1000

	print w2distance1D(mu1,sigma1,mu2,sigma2)
	print w2distance1D(mu1,sigma1, mu3,sigma3)
	print w2distance1D(mu2,sigma2, mu3, sigma3)

def generateRandomData(length = 15, width = 2):
	"""
	Test function.
	"""
	return np.random.randn(length, width) #(valLimit, (length, width))

def randomSampling(data, nclusters):
	"""
	Helper function for the k-Means clustering.
	Returns given number of random samples from the data.
	"""
	randomIndex = np.random.randint(0,data.shape[0], nclusters)
	return data[randomIndex, :]

def NDMean(X):
	"""
	A "somewhat" modified function to find the mean of normal distributions in a 
	given array of normal distributions
	"""
	m1 = 0
	m2 = 0
	for i in range(X.shape[0]):
		m1 += X[i,0]
		m2 += X[i,1]

	m1 = m1/X.shape[0]
	m2 = m2/X.shape[0]
	return np.array([m1,m2])


def kmeans(data, nclusters, niter,delta,datatype, verbose = False):

	"""
	Implementation of the kmeans algorithm.
	The k-Means can be deployed by using either mean or median values, of which only mean has been implemented in this version.

	Input:
			data = Either a matrix of 1 dimensional or 2 dimensional. Each row should contain a mean and variance values
			nclusters = Total number of clusters required. 
			niter = Total number of iterations for clustering. More iterations => More time
			delta = precision
			datatype = 1: One-Dimenesional
					   2: Two-Dimensional	
			verbose = Level of Verbosity

	Output: c = Cluster Centers
			x = Cluster Class
			d = Distance of all points from the cluster centers
	"""

	initial = randomSampling(data, nclusters)
	N, dim = data.shape
	k, cdim = initial.shape

	if dim != cdim:  
		raise ValueError("Error! Centers must have same number of columns as the data!")

	allX = np.arange(N)
	oldDist = 0

	for jiter in range(1, niter+1):
		if datatype == 1:
			dist = distanceFunction1D(data, initial)
		elif datatype == 2:
			dist = distanceFunction2D(data, initial)
		else:
			raise ValueError("Datatype Can Either be 1 or 2. 1:One-Dimenesional Normal Distribution, 2:Two-Dimensional Normal Distribution")
		xtoc = dist.argmin(axis = 1)
		distances = dist[allX, xtoc]
		avgdist = distances.mean()

		if (1-delta) * oldDist <= avgdist <= oldDist \
		or jiter==niter:
			break
		oldDist = avgdist

		for jc in range(k):
			c = np.where(xtoc == jc)[0]
			if len(c) > 0:
				initial[jc] = NDMean(data[c])#.mean(axis = 0)

	return initial, xtoc, distances 


if __name__ == '__main__':
	#testDistance1D()
	randomdata = generateRandomData(length = 1000, width = 2)

	# print "load image..."
	# image = readConvert('/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/crop.jpg')
	# OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 5)
	# mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)
	# #WInt1 = SScomputeCSWassersteinIntensity(mu_c_int, sig_c_int, mu_s_int, sig_s_int)
	# #print mu_c_int[0,0]
	# #plotImg(mu_c_int[0,0])
	# #print cND_Int_mu
	# mu_sigma = np.array([mu_c_int[0,0].ravel(), sig_c_int[0,0].ravel()]).T # np.asarray(zip(mu_c_int[0,0].ravel(), sig_c_int[0,0].ravel())) #splitcenterdata(cND_Int_mu, cND_Int_sigma)
	# randomdata = mu_sigma
	print randomdata
	print randomdata.shape
	#print randomSampling(generateRandomData(), 3)
	nclusters = 3
	c,x,d = kmeans(randomdata, nclusters,10,0.01,3, verbose = True)

	data = randomdata

	idx = x # vq(data,c)	

	centroids = c

	plot(data[idx==0,0],data[idx==0,1],'ob',
	     data[idx==1,0],data[idx==1,1],'or',
	     data[idx==2,0],data[idx==2,1],'og')#,
	     # data[idx==3,0],data[idx==3,1],'oy',
	     # data[idx==4,0],data[idx==4,1],'oc')

	plot(centroids[:,0],centroids[:,1],'sc',markersize=10)
	show()




