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

	#cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDColor(image)
	#WInt1 = computeCSWassersteinColor(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma)

	cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDIntensity(image)
	#WInt2 = computeCSWassersteinIntensity(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma)

	mu_sigma = splitcenterdata(cND_Int_mu, cND_Int_sigma)

	print mu_sigma[0]

	print mu_sigma[0].shape

	print len(mu_sigma)
	
	# clustering
	data = mu_sigma[0]
	centroids,_ = kmeans(data,5)

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