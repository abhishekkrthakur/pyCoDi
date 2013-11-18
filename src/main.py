#main.py

import numpy as np
import cv2
from ImageOperations import *
import pylab as pl
import matplotlib.cm as cm

def plotImg(image):
	pl.imshow(image, cmap = cm.gray)  
	pl.show()


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def plotND(mean, variance):
	sigma = np.sqrt(variance)
	x = np.linspace(-100,100,1000)
	plt.plot(x,mlab.normpdf(x,mean,sigma))
	plt.show()

if __name__ == '__main__':
	# load the image as RGB, NOT BGR
	print "load image..."
	image = readImg('../testimages/dscn4311.jpg')

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


	print "get gaussian for all supplimenting layers of intensity..."
	for i in range(len(iSup)):
		for j in range(len(iSup[i])):
			iSup[i][j] = csEstimate(iSup[i][j], 1.5)


	#print iSup[0][0].shape
	print "center normal distribution for all intensity layers..."	
	# plotImg(iSup[1][1])
	cND_Int_mu = []
	for i in range(len(iSup)):
		cND_Int_mu.append(iSup[i][0])

	cND_Int_sigma = []
	for i in range(len(iSup)):
		cND_Int_sigma.append(iSup[i][1] - np.multiply(iSup[i][0], iSup[i][0]))

	#plotImg(cND_Int_sigma[0])2

	# plotImg(cND_Int_mu[0][])

	#plotND(cND_Int_mu[1][0,0], cND_Int_sigma[1][0,0]) 
	#plotND(cND_Int_mu[4][0,0], cND_Int_sigma[4][0,0]) 

	print "done"









	# print "create supplimenting layers for color..."

	# cSup = []

	# for i in range(len(c1)):
	# 	colorSup = supplementing_layers_color(c1[i], c2[i])
	# 	cSup.append(colorSup)

	# print len(cSup)
	# print len(cSup[0])





	


