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
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from kMeansW2 import *
#from kMeansW2_2D import *


def savePlot(data, filename):
#Rescale to 0-255 and convert to uint8
	base = os.path.basename(filename)[:-4]
	head, tail = os.path.split(filename)
	rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
	im = Image.fromarray(rescaled)
	im.save(head + '/' + base + '.png')


def plotImg(data):
	rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
	cv2.imshow('plot of image', rescaled)
	cv2.waitKey(0)
	#pl.imshow(image, cmap = cm.gray) 
	#pl.tight_layout() 
	#pl.show()

def scaleSpaceRepresentation(image, scales, octaves):
	"""
	Returns a Octvave-Scale matrix
	Input: 3-Channel Image, number of scales, number of octaves
	Output: A matrix which contains all images for octaves and scales
			except the original image and original scales
	"""

	#Oct = []
	Oct = np.empty((octaves + 1, scales + 1), dtype = 'object')
	
	#print Oct
	Oct[0,0] = image
	tempimg = []
	for i in range(octaves+1):
		print i
		if i == 0: 
			tempimg = Oct[0,0]
		else:
			newimg = Oct[i-1,-1]
			Oct[i,0] = newimg[::2, ::2]

		for j in range(scales):
			#tempimg = Oct[i,0]
			#print Oct[i, j]
			gau_sigma = 2.0 ** ((j+1)/scales)
			ks = int(math.ceil(3*gau_sigma))
			if ks%2 == 0:
				ks += 1
			ksize = (ks,ks)
			Oct[i, j+1] = smoothImg(Oct[i,j], sigmaX = gau_sigma, sigmaY = gau_sigma, ksize  = ksize)
			
	return Oct#[1:,:-1]


def SS_supp_Intensity(OSMatrix):
	"""
	
	Create supplimenting layers for the intensity channel (every Octave and every scale!!) .
	Input: Octvave-Scale matrix
	Output: i, i^2 Matrices for every scale and every octave in the same format as the OSMatrix

	"""
	#print OSMatrix
	supInt1 = np.empty((OSMatrix.shape), dtype = 'object')
	supInt2 = np.empty((OSMatrix.shape), dtype = 'object')

	for i in range(OSMatrix.shape[0]):
		for j in range(OSMatrix.shape[1]):
			supInt1[i,j] = OSMatrix[i,j][:,:,0] 			# Taking only the intensity channel
			supInt2[i,j] = OSMatrix[i,j][:,:,0]**2.0 		# Take the square of all elements of the intensity channel

	return supInt1, supInt2

 # supInt1 = np.empty((OSMatrix.shape), dtype = 'object')
 #        supInt2 = np.empty((OSMatrix.shape), dtype = 'object')
 #        supInt1 =  OSMatrix[:,:,0] 
 #        supInt2 = OSMatrix[:,:,0]**2.0

def SS_supp_Color(OSMatrix):
	"""

	Create supplimenting layers for color channel (Every Octave and every scale is used!)
	Input : Octvave-Scale matrix
	Output : c1,c2,c1**2,c2**2, c1c2 for every scale and every octave in the same format as OSMatrix
	"""

	c1 = np.empty((OSMatrix.shape), dtype = 'object')
	c2 = np.empty((OSMatrix.shape), dtype = 'object')
	c1_2 = np.empty((OSMatrix.shape), dtype = 'object')
	c2_2 = np.empty((OSMatrix.shape), dtype = 'object')
	c1c2 = np.empty((OSMatrix.shape), dtype = 'object')

	for i in range(OSMatrix.shape[0]):
		#print i
		for j in range(OSMatrix.shape[1]):
			c1[i,j] = OSMatrix[i,j][:,:,1]
			c2[i,j] = OSMatrix[i,j][:,:,2]
			c1_2[i,j] = c1[i,j] ** 2.0 
			c2_2[i,j] = c2[i,j] ** 2.0
			c1c2[i,j] = c1[i,j] * c2[i,j]
			#print c1c2[i,j].shape

	#plotImg(c1[0,0])

	return c1,c2,c1_2,c2_2,c1c2


def SSCS_Dist_Intensity(OSMatrix, sizeIn, sizeOut):
	"""
	
	Creates center-surround matrix for all scales and octaves
	present in the Octvave-Scale matrix.

	params:
		OSMatrix = Octvave-Scale Matrix
		sizeIn = center Gaussian standard deviation
		sizeOut = surround Gaussian standard deviation
	
	Output: 

		mu_c_int		|
		sig_c_int		| All are of the same shape as OSMatrix
		mu_s_int		| All contain a single mean/variance value as they are one-dimensional
		sig_s_int		|

	"""

	# get the supplimenting layers first:
	supInt1, supInt2 = SS_supp_Intensity(OSMatrix)

	mu_c_int = np.empty((supInt1.shape), dtype = 'object')
	sig_c_int = np.empty((supInt2.shape), dtype = 'object')
	mu_s_int = np.empty((supInt1.shape), dtype = 'object')
	sig_s_int = np.empty((supInt2.shape), dtype = 'object')


	for i in range(supInt1.shape[0]):
		print i
		for j in range(supInt1.shape[1]):
			inten = supInt1[i,j]
			inten2 = supInt2[i,j]
			mu_c_int[i,j] = csEstimate(inten, sizeIn)
			mu_s_int[i,j] = csEstimate(inten, sizeOut)
			sig_c_int[i,j] = csEstimate(inten2, sizeIn) - np.multiply(csEstimate(inten,sizeIn), csEstimate(inten,sizeIn))
			#print sig_c_int[i,j]
			sig_s_int[i,j] = csEstimate(inten2, sizeOut) - np.multiply(csEstimate(inten,sizeOut), csEstimate(inten,sizeOut))

	return mu_c_int, sig_c_int, mu_s_int, sig_s_int


def SSCS_Dist_Color(OSMatrix, sizeIn, sizeOut):
	"""
	
	Creates center-surround matrix for all scales and octaves
	present in the Octvave-Scale matrix.

	params:
		OSMatrix = Octvave-Scale Matrix
		sizeIn = center Gaussian standard deviation
		sizeOut = surround Gaussian standard deviation
	
	Output: 

		mu_c_col		|
		sig_c_col		| All are of the same shape as OSMatrix
		mu_s_col		| All contain a matrix of mean/variance value as they are "2-Dimensional"
		sig_s_col		|

	"""

	c1,c2,c1_2,c2_2,c1c2 = SS_supp_Color(OSMatrix)

	mu_c_col = np.empty((OSMatrix.shape), dtype = 'object')
	mu_s_col = np.empty((OSMatrix.shape), dtype = 'object')
	sig_c_col = np.empty((OSMatrix.shape), dtype = 'object')
	sig_s_col = np.empty((OSMatrix.shape), dtype = 'object')


	# Find c1_bar, c2_bar, c1_2_bar, c2_2_bar, c1c2_bar for center and surround
	c1_bar_c = np.empty((OSMatrix.shape), dtype = 'object')
	c2_bar_c = np.empty((OSMatrix.shape), dtype = 'object')
	c1_2_bar_c = np.empty((OSMatrix.shape), dtype = 'object')
	c2_2_bar_c = np.empty((OSMatrix.shape), dtype = 'object')
	c1c2_bar_c = np.empty((OSMatrix.shape), dtype = 'object')

	c1_bar_s = np.empty((OSMatrix.shape), dtype = 'object')
	c2_bar_s = np.empty((OSMatrix.shape), dtype = 'object')
	c1_2_bar_s = np.empty((OSMatrix.shape), dtype = 'object')
	c2_2_bar_s = np.empty((OSMatrix.shape), dtype = 'object')
	c1c2_bar_s = np.empty((OSMatrix.shape), dtype = 'object')

	for i in range(OSMatrix.shape[0]):
		for j in range(OSMatrix.shape[1]):
			c1_bar_c[i,j] = csEstimate(c1[i,j], sizeIn)
			c1_bar_s[i,j] = csEstimate(c1[i,j], sizeOut)

			c2_bar_c[i,j] = csEstimate(c2[i,j], sizeIn)
			c2_bar_s[i,j] = csEstimate(c2[i,j], sizeOut)

			c1_2_bar_c[i,j] = csEstimate(c1_2[i,j], sizeIn)
			c1_2_bar_s[i,j] = csEstimate(c1_2[i,j], sizeOut)

			c2_2_bar_c[i,j] = csEstimate(c2_2[i,j], sizeIn)
			c2_2_bar_s[i,j] = csEstimate(c2_2[i,j], sizeOut)

			c1c2_bar_c[i,j] = csEstimate(c1c2[i,j], sizeIn)
			c1c2_bar_s[i,j] = csEstimate(c1c2[i,j], sizeOut)
	# plotImg(c1_bar_c[0,0])
	# plotImg(c1_bar_c[0,1])
	# plotImg(c1_bar_c[1,0])
	# plotImg(c1_bar_c[1,1])

	#print c1_bar_c[0,0]
	#print c1_bar_c[0,1]

	# create mu for center and surround


	for i in range(OSMatrix.shape[0]):
		for j in range(OSMatrix.shape[1]):
			temparr1 = np.empty((c1[i,j].shape), dtype = 'object')
			temparr2 = np.empty((c1[i,j].shape), dtype = 'object')

			for p in range(c1[i,j].shape[0]):
				for q in range(c2[i,j].shape[1]):
					#print c1_bar_c[i,j][p,q]
					#print c1_bar_c[i,j][p,q],c2_bar_c[i,j][p,q]
					tempmu1 = np.zeros((2,1))
					tempmu2 = np.zeros((2,1))
					
					tempmu1[0,0] = c1_bar_c[i,j][p,q]
					tempmu1[1,0] = c2_bar_c[i,j][p,q]
					tempmu2[0,0] = c1_bar_s[i,j][p,q]
					tempmu2[1,0] = c2_bar_s[i,j][p,q]

					temparr1[p,q] = tempmu1
					temparr2[p,q] = tempmu2
					#print tempmu1, tempmu2

			mu_c_col[i,j] = temparr1
			mu_s_col[i,j] = temparr2
			#print mu_c_col[i,j]

	# create sigma for center and surround


	for i in range(OSMatrix.shape[0]):
		for j in range(OSMatrix.shape[1]):
			temparr1 = np.empty((c1[i,j].shape), dtype = 'object')
			temparr2 = np.empty((c1[i,j].shape), dtype = 'object')

			for p in range(c1[i,j].shape[0]):
				for q in range(c2[i,j].shape[1]):
					#print c1_bar_c[i,j][p,q]
					#print c2_bar_c[i,j][p,q]
					tempmu1 = np.zeros((2,2))
					tempmu2 = np.zeros((2,2))
					
					tempmu1[0,0] = c1_2_bar_c[i,j][p,q] - (c1_bar_c[i,j][p,q] ** 2.0)
					tempmu1[0,1] = c1c2_bar_c[i,j][p,q] - (c1_bar_c[i,j][p,q] * c2_bar_c[i,j][p,q])
					tempmu1[1,0] = c1c2_bar_c[i,j][p,q] - (c1_bar_c[i,j][p,q] * c2_bar_c[i,j][p,q])
					tempmu1[1,1] = c2_2_bar_c[i,j][p,q] - (c2_bar_c[i,j][p,q] ** 2.0)

					tempmu2[0,0] = c1_2_bar_s[i,j][p,q] - (c1_bar_s[i,j][p,q] ** 2.0)
					tempmu2[0,1] = c1c2_bar_s[i,j][p,q] - (c1_bar_s[i,j][p,q] * c2_bar_s[i,j][p,q])
					tempmu2[1,0] = c1c2_bar_s[i,j][p,q] - (c1_bar_s[i,j][p,q] * c2_bar_s[i,j][p,q])
					tempmu2[1,1] = c2_2_bar_s[i,j][p,q] - (c2_bar_s[i,j][p,q] ** 2.0)

					temparr1[p,q] = tempmu1
					temparr2[p,q] = tempmu2

			sig_c_col[i,j] = temparr1
			sig_s_col[i,j] = temparr2

	return mu_c_col, sig_c_col, mu_s_col, sig_s_col


def normalize(arr):
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()

        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr

def plot1DND(mean, variance):
	sigma = np.sqrt(variance)
	mu = mean
	s = np.random.normal(mu, sigma, 1000)
	count, bins, ignored = plt.hist(s, 30, normed=True)
	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
	plt.show()

def plot2DND(mean, variance):
	mean1 = mean.flatten()
	cov1 = variance

	nobs = 2500
	rvs1 = np.random.multivariate_normal(mean1, cov1, size=nobs)

	plt.plot(rvs1[:, 0], rvs1[:, 1], '.')
	plt.axis('equal')
	plt.show()


def SScomputeCSWassersteinIntensity(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma):
	
	WInt = np.empty((cND_Int_mu.shape), dtype = 'object')
	for i in range(cND_Int_mu.shape[0]):
		for j in range(cND_Int_mu.shape[1]):
			tempimg = np.zeros((cND_Int_mu[i,j].shape))
			for p in range(cND_Int_mu[i,j].shape[0]):
				for q in range(cND_Int_mu[i,j].shape[1]):
					t1 = np.linalg.norm(cND_Int_mu[i,j][p,q]-sND_Int_mu[i,j][p,q])
					t1 = t1 * t1
					t2 = sND_Int_sigma[i,j][p,q] + cND_Int_sigma[i,j][p,q] 
					p1 = cND_Int_sigma[i,j][p,q]
					p2 = cND_Int_sigma[i,j][p,q]
					p3 = sND_Int_sigma[i,j][p,q]

					if p1 < 0.0:
						p1 = 0.0

					if p2 < 0.0:
						p2 = 0.0

					if p3 < 0.0:
						p3 = 0.0

					t3 = 2.0 * np.sqrt(np.sqrt(p1) * p3 * np.sqrt(p2))
					if (t1 + t2 - t3) < 0:
						tempimg[p,q] = 0.0
					else:
						tempimg[p,q] = np.sqrt(t1 + t2 - t3)

			WInt[i,j] = tempimg

	return WInt


def SScomputeCSWassersteinColor(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma):
	WInt = np.empty((cND_Int_mu.shape), dtype = 'object')
	#print (cND_Int_mu[0].shape), (cND_Int_sigma[0].shape), (sND_Int_mu[0].shape), (sND_Int_sigma[0].shape)
	for i in range(cND_Int_mu.shape[0]):
		for j in range(cND_Int_mu.shape[1]):
			print i
			#print "shapeeeeeee", cND_Int_mu[0][0].shape
			tempimg = np.zeros((cND_Int_mu[i,j].shape))
			for p in range(cND_Int_mu[i,j].shape[0]):
				for q in range(cND_Int_mu[i,j].shape[1]):
					#print (i,j,k)
					#print cND_Int_mu[i,j][p,q].shape
					#print cND_Int_mu[i,j][p,q], sND_Int_mu[i,j][p,q]
					t1 = np.linalg.norm(cND_Int_mu[i,j][p,q]-sND_Int_mu[i,j][p,q])

					#print t1
					t1 = t1 ** 2.0
					#print t1
					t2 = np.trace(sND_Int_sigma[i,j][p,q]) + np.trace(cND_Int_sigma[i,j][p,q]) 
					p1 = np.trace(np.dot(cND_Int_sigma[i,j][p,q], sND_Int_sigma[i,j][p,q]))
					p2 =  (((np.linalg.det(np.dot(cND_Int_sigma[i,j][p,q], sND_Int_sigma[i,j][p,q])))))
					if p2 < 0.0:
						p2 = 0.0
					p2 = np.sqrt(p2)
					tt = p1 + 2.0*p2
					if tt < 0.0:
						tt = 0.0
					t3 = 2.0 * np.sqrt(tt)
					#print t3
					if (t1 + t2 - t3) < 0:
						tempimg[p,q] = 0.0
						#print "here"
					else:
						tempimg[p,q] = np.sqrt(t1 + t2 - t3)
					#print tempimg[j,k]

			WInt[i,j] = tempimg
	return WInt



def SScombineScales(WInt):


	origshape1 = WInt[0,0].shape[1]
	origshape0 = WInt[0,0].shape[0]
	origshape = WInt[0,0].shape

	for i in range(WInt.shape[0]):
		for j in range(WInt.shape[1]):
			temp = WInt[i,j]
			if temp.shape == origshape:
				continue
			else:
				WInt[i,j] = cv2.resize(WInt[i,j], (origshape1, origshape0), interpolation=cv2.INTER_CUBIC)

	tempimg = np.zeros((origshape))
	for i in range(WInt.shape[0]):
		for j in range(WInt.shape[1]):
			tempimg += np.sqrt(i+1) * WInt[i,j]

	tempimg = tempimg/(origshape0 * origshape1)

	return tempimg 

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

def cropTest(mu_c_int, sig_c_int, mu_s_int, sig_s_int, left, upper, right, lower):

	"""

	Crops the center and surround distributions from all scales and octaves

	"""

	mu_c = np.empty((mu_s_int.shape), dtype = 'object')
	sig_c = np.empty((mu_s_int.shape), dtype = 'object')
	mu_s = np.empty((mu_s_int.shape), dtype = 'object') 
	sig_s = np.empty((mu_s_int.shape), dtype = 'object')

	#print mu_c_int[0,0][2:4,:-1].shape

	#26 55 59 140
	#55:140, 26:59
	#(170, 82)
	for i in range(mu_c_int.shape[0]):
		le = int(left/(2**i))
		up = int(upper/(2**i))
		rt = int(right/(2**i))
		lo = int(lower/(2**i))
		
		#print i
		for j in range(mu_c_int.shape[1]):
			#print mu_c_int[i,j].shape
			#print le,up,rt,lo

			##plotImg(mu_c_int[i,j])
			mu_c[i,j] = mu_c_int[i,j][up:lo,le:rt]
			sig_c[i,j] = sig_c_int[i,j][up:lo,le:rt]
			mu_s[i,j] = mu_s_int[i,j][up:lo,le:rt]
			sig_s[i,j] = sig_s_int[i,j][up:lo,le:rt]

	return mu_c, sig_c, mu_s, sig_s

def kMeansInt(mu_c, sig_c, n_iter = 100, n_clusters = 3, delta = 0.001):
	"""
	mu_c and sig_c have the same shape as OSMatrix.
	mu_c and sig_c are the cropped mu and sigma for every region of the OSMatrix
	"""
	centroids = np.empty((mu_c.shape), dtype = 'object')

	for i in range(mu_c.shape[0]):
		for j in range(mu_c.shape[1]):
			mu_sigma = np.asarray(zip(mu_c_int[i,j].ravel(), sig_c_int[i,j].ravel()))
			data = mu_sigma

			X = data
			ncluster = n_clusters
			kmdelta = delta
			kmiter = n_iter

			randomcentres = randomsample1d( X, ncluster )

			centres, xtoc, dist = kmeans1d( X, randomcentres,
										delta=kmdelta, maxiter=kmiter, verbose=2)

			centroids[i,j] = centres

	return centroids

def kMeansCol(mu_c, sig_c, n_iter = 100, n_clusters = 3, delta = 0.001):
	"""
	mu_c and sig_c have the same shape as OSMatrix.
	mu_c and sig_c are the cropped mu and sigma for every region of the OSMatrix
	"""
		



if __name__ == '__main__':
	image = readConvert('/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/crop.jpg')
	print image.shape
	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 5)
	#print (OSMatrix[0,0].shape)
	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)

	#print (mu_c_int[0,0][1,1], mu_s_int[0,0][1,1])
	#print (mu_c_int[0,0][2,2], mu_s_int[0,0][2,2])

	WInt1 = SScomputeCSWassersteinIntensity(mu_c_int, sig_c_int, mu_s_int, sig_s_int)
	

	mu_c_col, sig_c_col, mu_s_col, sig_s_col = SSCS_Dist_Color(OSMatrix, 1.0, 10.0)

	#print (mu_c_int[0,0][1,1], mu_s_int[0,0][1,1])
	#print (mu_c_int[0,0][2,2], mu_s_int[0,0][2,2])

	# WInt2 = SScomputeCSWassersteinColor(mu_c_col, sig_c_col, mu_s_col, sig_s_col)

	WInt1 = SScombineScales(WInt1)
	# WInt2 = SScombineScales(WInt2)

	# fin  = (WInt1 + WInt2)/2.0

	# plotImg(fin)
	plotImg(WInt1)
	# plotImg(WInt2)
	# plotImg(WInt[0,1])
	# plotImg(WInt[0,2])
	# plotImg(WInt[1,0])
	# plotImg(WInt[1,1])
	# plotImg(WInt[1,2])
	#print mu_c_int[0,0][0,0], sig_c_int[0,0][0,0]
	
	#plot1DND(mu_c_int[0,0][0,1], sig_c_int[0,0][0,1])
	