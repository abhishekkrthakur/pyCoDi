"""

Clustering for training data using k-means/APC

__author__ : Abhishek Thakur

"""


import numpy as np
import cv2
from ImageOperations import *
import pylab as pl
import copy
import matplotlib.cm as cm
from cv2 import *
from scipy.linalg import sqrtm
from skimage.transform.pyramids import pyramid_laplacian
import sys
import Image
import os
from utils import *
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

def W2_center_center_Intensity(arr1,arr2):
	arr1 = np.asarray(arr1)
	
	dist_fin = []
	for i in range(arr1.shape[0]):
		#mu1 = arr1[i][0]
		distance = []
		#print arr1[i]
		#sig1 = arr1[i][1]
		for j in range(arr2.shape[0]):
			#mu2 = arr2[j][0]
			#sig2 = arr2[j][1]

			dist = np.linalg.norm(np.asarray(arr1[i,:])-np.asarray(arr2[j,:]))
			distance.append(dist)
		dist_fin.append(np.asarray(distance))

	return dist_fin



def splitcenterdata(cND_Int_mu, cND_Int_sigma):
	mu_sigma = []
	for i in range(len(cND_Int_mu)):
		#temparr = np.zeros((2,cND_Int_mu[i].shape[0]*cND_Int_mu[i].shape[1]))
		temparr1 = np.reshape(cND_Int_mu[i],(-1,1))
		temparr2 = np.reshape(cND_Int_sigma[i],(-1,1))
		temparr = np.column_stack((temparr1, temparr2))
		mu_sigma.append(temparr)

	return mu_sigma

if __name__ == '__main__':
	# load the image as RGB, NOT BGR
	print "load image..."
	try:
		filename = sys.argv[1]
		
	except:
		print "no filename entered"

	image = readImg(filename)
	#print image

	#cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDColor(image)
	#WInt1 = computeCSWassersteinColor(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma)

	cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDIntensity(image)
	#WInt2 = computeCSWassersteinIntensity(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma)
	#print cND_Int_mu
	mu_sigma = splitcenterdata(cND_Int_mu, cND_Int_sigma)

	# print mu_sigma[0]

	# print mu_sigma[0].shape

	# print len(mu_sigma)
	
	# clustering
	data = mu_sigma[0]
	centroids,_ = kmeans(data,5)
	print centroids
	#print data
	# assign each sample to a cluster
	idx,_ = vq(data,centroids)

	# some plotting using numpy's logical indexing
	plot(data[idx==0,0],data[idx==0,1],'ob',
	     data[idx==1,0],data[idx==1,1],'or',
	     data[idx==2,0],data[idx==2,1],'og',
	     data[idx==3,0],data[idx==3,1],'oy',
	     data[idx==4,0],data[idx==4,1],'oc')

	plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
	show()

	# compare with full image

	image = readImg('/Users/abhishek/Documents/Thesis/testimages-saliency/dscn4311.jpg')

	#cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDColor(image)
	#WInt1 = computeCSWassersteinColor(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma)

	cND_Int_mu1, cND_Int_sigma1, sND_Int_mu, sND_Int_sigma = csNDIntensity(image)

	#plotImg(cND_Int_mu1[0])

	mu_sigma2 = splitcenterdata(cND_Int_mu1, cND_Int_sigma1)

	dist = W2_center_center_Intensity(centroids, mu_sigma2[0])

	# print dist[1].shape
	# print dist[2].shape
	# print dist[3].shape
	# print dist[4].shape

	plotImg(np.reshape(dist[0], cND_Int_mu1[0].shape))
	plotImg(np.reshape(dist[1], cND_Int_mu1[0].shape))
	plotImg(np.reshape(dist[2], cND_Int_mu1[0].shape))
	plotImg(np.reshape(dist[3], cND_Int_mu1[0].shape))
	plotImg(np.reshape(dist[4], cND_Int_mu1[0].shape))


	#print len(dist), mu_sigma2[0].shape, len(centroids)

