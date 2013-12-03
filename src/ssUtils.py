from __future__ import division
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
from ssUtils import *
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.cluster.vq import vq
import numpy as np
from ImageOperations import *
import pylab as pl
import copy
import matplotlib.cm as cm
import random
import sys
from time import time
from pylab import plot,show
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
import random
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
			
	return Oct[:,:-1]


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

def distanceFunction2D(X,Y):
	dist = np.zeros((X.shape[0], Y.shape[0]))
	#print X.shape, Y.shape
	for i in range(X.shape[0]):
		for j in range(Y.shape[0]):
			#print w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1])
			dist[i,j] = (w2distance2D(X[i,0], X[i,1], Y[j,0], Y[j,1]))

	return dist


def distanceFunction1D(X,Y):
	dist = np.zeros((X.shape[0], Y.shape[0]))
	#print X.shape, Y.shape
	for i in range(X.shape[0]):
		for j in range(Y.shape[0]):
			#print w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1])
			dist[i,j] = (w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1]))

	return dist


def kmeans2D( X, centres, delta=.001, maxiter=10, p=2, verbose=0 ):
    """ centres, Xtocentre, distances = kmeans2D( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeans2Dsample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    print X.shape
    if dim != cdim:
        raise ValueError( "kmeans2D: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print "kmeans2D: X %s  centres %s  delta=%.2g  maxiter=%d  " % (
            X.shape, centres.shape, delta, maxiter)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = distanceFunction2D(X, centres) #cdist_sparse( X, centres, metric=metric, p=p )
        #print D.shape, X.shape, centres.shape  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print "kmeans2D: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans2D: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans2D: cluster 50 % radius", r50.astype(int)
        print "kmeans2D: cluster 90 % radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

def kmeans2Dsample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans2D, fast for large N:
        1) kmeans2D a random sample of nsample ~ sqrt(N) from X
        2) full kmeans2D, starting from those centres
    """
        # merge w kmeans2D ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans2D( Xsample, pass1centres, **kwargs )[0]
    return kmeans2D( X, samplecentres, **kwargs )

def kmeans1D( X, centres, delta=.001, maxiter=10, p=2, verbose=0 ):
    """ centres, Xtocentre, distances = kmeans1D( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeans1Dsample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    #print X.shape
    if dim != cdim:
        raise ValueError( "kmeans1D: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print "kmeans1D: X %s  centres %s  delta=%.2g  maxiter=%d  " % (
            X.shape, centres.shape, delta, maxiter)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = distanceFunction1D(X, centres) #cdist_sparse( X, centres, metric=metric, p=p )
        #print D.shape, X.shape, centres.shape  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print "kmeans1D: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans1D: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans1D: cluster 50 % radius", r50.astype(int)
        print "kmeans1D: cluster 90 % radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

def kmeans1Dsample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans1D, fast for large N:
        1) kmeans1D a random sample of nsample ~ sqrt(N) from X
        2) full kmeans1D, starting from those centres
    """
        # merge w kmeans1D ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans1D( Xsample, pass1centres, **kwargs )[0]
    return kmeans1D( X, samplecentres, **kwargs )



def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    

    return d

def randomsample( X, n ):
    """ 
    
    random.sample of the rows of X
    X may be sparse -- best csr

    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcentres( X, centres, p=2 ):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()


def kMeansInt(mu_c, sig_c, n_iter = 100, n_clusters = 3, delta = 0.001, verbose = 2):
	"""
	mu_c and sig_c have the same shape as OSMatrix.
	mu_c and sig_c are the cropped mu and sigma for every region of the OSMatrix
	"""
	centroids = np.empty((mu_c.shape), dtype = 'object')

	#print mu_c[0,0].shape, mu_c[1,0].shape, mu_c[2,0].shape

	for i in range(mu_c.shape[0]):
		for j in range(mu_c.shape[1]):
			mu_sigma = np.asarray(zip(mu_c[i,j].ravel(), sig_c[i,j].ravel()))
			#print mu_sigma.shape
			data = mu_sigma

			X = data
			ncluster = n_clusters
			kmdelta = delta
			kmiter = n_iter

			if X.shape[0] <= ncluster:
				ncluster = 1

			randomcentres = randomsample( X, ncluster )

			centres, xtoc, dist = kmeans1D( X, randomcentres,
										delta=kmdelta, maxiter=kmiter, verbose=verbose)

			centroids[i,j] = centres

	return centroids

def kMeansCol(mu_c, sig_c, n_iter = 100, n_clusters = 3, delta = 0.001, verbose = 2):
	"""
	mu_c and sig_c have the same shape as OSMatrix.
	mu_c and sig_c are the cropped mu and sigma for every region of the OSMatrix
	"""
	centroids = np.empty((mu_c.shape), dtype = 'object')

	for i in range(mu_c.shape[0]):
		for j in range(mu_c.shape[1]):
			mu_sigma = np.array([mu_c[i,j].ravel(), sig_c[i,j].ravel()]).T

			data = mu_sigma
			X = data
			ncluster = n_clusters
			kmdelta = delta
			kmiter = n_iter

			if X.shape[0] <= ncluster:
				ncluster = 1

			randomcentres = randomsample( X, ncluster )

			centres, xtoc, dist = kmeans2D( X, randomcentres,
					delta=kmdelta, maxiter=kmiter, verbose=verbose)

			centroids[i,j] = centres

	return centroids


# def computeCentroidDistInt(numpyarray, centroids):
# 	"""
	
# 	2d numpy array with mu and sigma values,
# 	All centroid with mu and sigma after clustering

# 	"""
# 	newarr = np.zeros((numpyarray.shape))
# 	for i in range(numpyarray.shape[0]):
# 		for j in range(numpyarray.shape[1]):
# 			tempval = []
# 			print len(tempval)
# 			for p in range(centroids[i,j].shape[0]):
# 				tempval.append(w2distance1D(centroids[] ))
# 				newarr[i,j] = w2distance1D(centroid)


def computeW2CentroidDiffInt(centroids, OSMatrixTestmu, OSMatrixTestsigma ):
#w2distance1D(mu1, sig1, mu2, sig2)

	tempmat = np.empty((OSMatrixTestmu.shape), dtype = 'object')

	for i in range(OSMatrixTestmu.shape[0]):
		print i
		for j in range(OSMatrixTestmu.shape[1]):
			tempimg = np.zeros((OSMatrixTestmu[i,j].shape))
			for r in range(OSMatrixTestmu[i,j].shape[0]):
				for s in range(OSMatrixTestmu[i,j].shape[1]):
					lencent = centroids[i,j].shape[0]
					#print lencent
					dist = []
					for p in range(lencent):
						dist.append(w2distance1D(OSMatrixTestmu[i,j][r,s], OSMatrixTestsigma[i,j][r,s],centroids[i,j][p,0],centroids[i,j][p,1]))
					#print dist
					val = np.exp(-np.min(dist))# - np.min(dist)
					#print val
					tempimg[r,s] = val
			tempmat[i,j] = tempimg

	return tempmat

def computeW2CentroidDiffCol(centroids, OSMatrixTestmu, OSMatrixTestsigma ):
#w2distance1D(mu1, sig1, mu2, sig2)

	tempmat = np.empty((OSMatrixTestmu.shape), dtype = 'object')

	for i in range(OSMatrixTestmu.shape[0]):
		print i
		for j in range(OSMatrixTestmu.shape[1]):
			tempimg = np.zeros((OSMatrixTestmu[i,j].shape))
			for r in range(OSMatrixTestmu[i,j].shape[0]):
				for s in range(OSMatrixTestmu[i,j].shape[1]):
					lencent = centroids[i,j].shape[0]
					#print lencent
					dist = []
					for p in range(lencent):
						dist.append(w2distance2D(OSMatrixTestmu[i,j][r,s], OSMatrixTestsigma[i,j][r,s],centroids[i,j][p,0],centroids[i,j][p,1]))
					#print dist
					val = np.exp(-np.min(dist))
					#print val
					tempimg[r,s] = val
			tempmat[i,j] = tempimg

	return tempmat


def invertImg(image):

	im = Image.fromarray(np.uint8(image))
	#plotImg(np.array(im))
	inverted = Image.eval(im, lambda(x):255-x)
	inverted = np.array(inverted)
	#plotImg(inverted)
	return inverted



if __name__ == '__main__':
	image = readConvert('/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/crop.jpg')
	print image.shape
	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 3)
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
	