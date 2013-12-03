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
	image = readConvert('/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/crop.jpg')
	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 5)
	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)
	#WInt1 = SScomputeCSWassersteinIntensity(mu_c_int, sig_c_int, mu_s_int, sig_s_int)
	
	#print cND_Int_mu
	mu_sigma = np.asarray(zip(mu_c_int[0,0].ravel(), sig_c_int[0,0].ravel())) #splitcenterdata(cND_Int_mu, cND_Int_sigma)

	# print mu_sigma[0]

	# print mu_sigma[0].shape

	# print len(mu_sigma)
	
	# clustering
	data = mu_sigma
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

	