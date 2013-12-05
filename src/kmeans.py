# k-Means clustering for Normal Distributions - Almost from scratch!

import numpy as np
import scipy as sp
from ssUtils import *
from ImageOperations import *
import random

def w2distance1D(mu1, sig1, mu2, sig2):
	"""

	Return the one dimensional W2 distance on euclidean norm.
	all mu and sigma values are one dimensional single values

	"""

	t1 = np.linalg.norm(mu1 - mu2)
	t1 = t1 * t1
	t2 = sig1 + sig2
	p1 = sig1
	p2 = sig1
	p3 = sig2

	if p1 < 0.0:
		p1 = 0.0

	if p2 < 0.0:
		p2 = 0.0

	if p3 < 0.0:
		p3 = 0.0

	t3 = 2.0 * np.sqrt(np.sqrt(p1) * p3 * np.sqrt(p2))
	if (t1 + t2 - t3) < 0:
		result = 0.0
	else:
		result = np.sqrt(t1 + t2 - t3)


	return result

def w2distance2D(mu1, sig1, mu2, sig2):
	"""

	Returns the Wasserstein distance between two 2-Dimensional normal distributions

	"""
	t1 = np.linalg.norm(mu1 - mu2)

	#print t1
	t1 = t1 ** 2.0
	#print t1
	t2 = np.trace(sig2) + np.trace(sig1) 
	p1 = np.trace(np.dot(sig1, sig2))
	p2 =  (((np.linalg.det(np.dot(sig1, sig2)))))
	if p2 < 0.0:
		p2 = 0.0
	p2 = np.sqrt(p2)
	tt = p1 + 2.0*p2
	if tt < 0.0:
		tt = 0.0
	t3 = 2.0 * np.sqrt(tt)
	#print t3
	if (t1 + t2 - t3) < 0:
		result = 0.0
		#print "here"
	else:
		result = np.sqrt(t1 + t2 - t3)

	return result

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

def randomSampling2D(data, nclusters):
	"""
	Helper function for the k-Means clustering.
	Returns given number of random samples from the data.
	"""
	dataarr = np.empty((nclusters, data.shape[1]), dtype = 'object')
	randomIndex = np.random.randint(0,data.shape[0], nclusters)
	#randomIndex = list(randomIndex)
	for i in range(len(randomIndex)):
		dataarr[i,:] = data[randomIndex[i]]
	return dataarr

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
	#print np.array([m1,m2]).shape
	return np.array([m1,m2])

def NDMean2D(X):
	"""
	A "somewhat" modified function to find the mean of normal distributions in a 
	given array of normal distributions
	"""
	m1 = 0
	m2 = 0
	meanarr = np.empty((2,), dtype = 'object')
	meanlist = []
	for i in range(X.shape[0]):
		m1 += X[i,0]
		m2 += X[i,1]

	m1 = m1/X.shape[0]
	m2 = m2/X.shape[0]
	
	#print m1,m2
	# for i in range(len(meanlist)):
	# 	meanarr[i,:] = np.asarray(meanlist[i])

	#print m2
	meanarr[0] = m1
	meanarr[1] = m2
	return meanarr


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

	
	if datatype == 1: 
		initial = randomSampling(data, nclusters)
	elif datatype == 2:
		initial = randomSampling2D(data, nclusters)
	else:
		raise ValueError("Datatype Can Either be 1 or 2. 1:One-Dimenesional Normal Distribution, 2:Two-Dimensional Normal Distribution")


	

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
				#if datatype == 2: print data[c].shape
				if datatype == 1:
					initial[jc] = NDMean(data[c])#.mean(axis = 0)
				elif datatype == 2:
					print initial[jc].shape, NDMean2D(data[c]).shape
					initial[jc] = NDMean2D(data[c])

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




