#main.py

import numpy as np
import cv2
from ImageOperations import *
import pylab as pl
import copy
import matplotlib.cm as cm
from cv2 import *

def plotImg(image):
	pl.imshow(image, cmap = cm.gray)  
	pl.show()


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def plotND(mean, variance):
	sigma = np.sqrt(variance)
	x = np.linspace(-10,10,1000)
	plt.plot(x,mlab.normpdf(x,mean,sigma))
	plt.show()

def csNDIntensity(image):
	print "convert colorspace..."
	l1 = convertColorspace(image) 

	print "create DOG pyramid..."
	L,c1,c2 = getDOGPyramid(l1, level=5, sigmaX=1.2,sigmaY=1.0,ksize=(5,5))


	print "create supplimenting layers for intensity..."


	iSup = []
	for i in range(len(L)):
		intensitySup = supplementing_layers_intensity(L[i])
		iSup.append(intensitySup)


	print len(iSup)

	# plotImg(iSup[1][1])
	icSup = copy.deepcopy(iSup)
	isSup = copy.deepcopy(iSup)

	print "get center gaussian for all supplimenting layers of intensity..."
	for i in range(len(iSup)):
		for j in range(len(iSup[i])):
			icSup[i][j] = csEstimate(iSup[i][j], 1.0)

	#print icSup[0][0][0,0], iSup[0][0][0,0], isSup[0][0][0,0]

	#print iSup[0][0].shape
	print "center normal distribution for all intensity layers..."	
	# plotImg(iSup[1][1])
	cND_Int_mu = []
	for i in range(len(icSup)):
		cND_Int_mu.append(icSup[i][0])

	cND_Int_sigma = []
	for i in range(len(icSup)):
		cND_Int_sigma.append(icSup[i][1] - np.multiply(icSup[i][0], icSup[i][0]))



	print "get surround gaussian for all supplimenting layers of intensity..."
	for i in range(len(iSup)):
		for j in range(len(iSup[i])):
			isSup[i][j] = csEstimate(iSup[i][j], 2.0)

	#print icSup[0][0][0,0], iSup[0][0][0,0], isSup[0][0][0,0]

	#print iSup[0][0].shape
	print "surround normal distribution for all intensity layers..."	
	# plotImg(iSup[1][1])
	sND_Int_mu = []
	for i in range(len(isSup)):
		sND_Int_mu.append(isSup[i][0])

	sND_Int_sigma = []
	for i in range(len(isSup)):
		sND_Int_sigma.append(isSup[i][1] - np.multiply(isSup[i][0], isSup[i][0]))


	return cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma




if __name__ == '__main__':
	# load the image as RGB, NOT BGR
	print "load image..."
	image = readImg('../testimages/dscn4311.jpg')



	cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDIntensity(image)

	a = zip(cND_Int_mu[0], cND_Int_sigma[0])
	b = zip(sND_Int_mu[0], sND_Int_sigma[0])

	#print (cND_Int_mu[0].shape)
	print cv.CalcEMD2(cND_Int_mu[0],cND_Int_mu[0],cv.CV_DIST_L2)

	#plotImg(sND_Int_sigma[0])

	# plotImg(cND_Int_mu[0][])

	#plotND(sND_Int_mu[4][0,0], sND_Int_sigma[4][0,0]) 
	#plotND(cND_Int_mu[4][0,0], cND_Int_sigma[4][0,0]) 


	# print "create supplimenting layers for color..."

	# cSup = []

	# for i in range(len(c1)):
	# 	colorSup = supplementing_layers_color(c1[i], c2[i])
	# 	cSup.append(colorSup)

	# print len(cSup)
	# print len(cSup[0])





	


