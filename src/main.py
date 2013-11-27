#main.py

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


if __name__ == '__main__':
	# load the image as RGB, NOT BGR
	print "load image..."
	try:
		filename = sys.argv[1]
		
	except:
		print "no filename entered"

	image = readImg(filename)

	cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDColor(image)
	WInt1 = computeCSWassersteinColor(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma)

	cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma = csNDIntensity(image)
	WInt2 = computeCSWassersteinIntensity(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma)

	s1 = combineScales(WInt1)
	s2 = combineScales(WInt2)

	s = (s1 + s2)/2.0
	plotImg(s1)
	plotImg(s2)
	plotImg(s)
	savePlot(s, filename)



	


