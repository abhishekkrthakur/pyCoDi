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

#function from David Cournapeau original scikits.learn em code
# To plot a confidence ellipse from multi-variate gaussian pdf
def gauss_ell(mu, va, dim = [0, 1], npoints = 100, level = 0.39):
    """ Given a mean and covariance for multi-variate
    gaussian, returns npoints points for the ellipse
    of confidence given by level (all points will be inside
    the ellipsoides with a probability equal to level)
    
    Returns the coordinate x and y of the ellipse"""
    
    c       = np.array(dim)

    if mu.size < 2:
        raise RuntimeError("this function only make sense for dimension 2 and more")

    if mu.size == va.size:
        mode    = 'diag'
    else:
        if va.ndim == 2:
            if va.shape[0] == va.shape[1]:
                mode    = 'full'
            else:
                raise DenError("variance not square")
        else:
            raise DenError("mean and variance are not dim conformant")

    # If X ~ N(mu, va), then [X` * va^(-1/2) * X] ~ Chi2
    chi22d  = stats.chi2(2)
    mahal   = np.sqrt(chi22d.ppf(level))
    
    # Generates a circle of npoints
    theta   = np.linspace(0, 2 * np.pi, npoints)
    circle  = mahal * np.array([np.cos(theta), np.sin(theta)])

    # Get the dimension which we are interested in:
    mu  = mu[dim]
    if mode == 'diag':
        va      = va[dim]
        elps    = np.outer(mu, np.ones(npoints))
        elps    += np.dot(np.diag(np.sqrt(va)), circle)
    elif mode == 'full':
        va  = va[c,:][:,c]
        #print "va = ", v a
        # Method: compute the cholesky decomp of each cov matrix, that is
        # compute cova such as va = cova * cova' 
        # WARN: scipy is different than matlab here, as scipy computes a lower
        # triangular cholesky decomp: 
        #   - va = cova * cova' (scipy)
        #   - va = cova' * cova (matlab)
        # So take care when comparing results with matlab !
        cova    = np.linalg.cholesky(va)
        elps    = np.outer(mu, np.ones(npoints))
        elps    += np.dot(cova, circle)
    else:
        raise DenParam("var mode not recognized")

    return elps[0, :], elps[1, :]


#from sklearn gmm
def make_ellipses(bvn, ax, level=0.95):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(bvn.covars[n][:2, :2])
        print v, w
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1]/u[0])
        angle = 180 * angle / np.pi # convert to degrees
        #v *= 39#25#9
        v = 2 * np.sqrt(v * stats.chi2.ppf(level, 2)) #JP
        ell = mpl.patches.Ellipse(bvn.mean[n, :2], v[0], v[1], 180 + angle,
                                  facecolor='none', 
                                  edgecolor=None, #color,
                                  ls='dashed',
                                  lw=3)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        
class BVN(object):
    pass

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
			
			#tempimg = Oct[i, j+1]

	#print Oct[2,0]
	# plotImg(Oct[0,0][:,:,0])
	# plotImg(Oct[0,1])
	# plotImg(Oct[0,2])
	# plotImg(Oct[0,3])
	# plotImg(Oct[1,0])
	# plotImg(Oct[1,1])
	# plotImg(Oct[1,2])
	# plotImg(Oct[1,3])
	# plotImg(Oct[2,0])
	# plotImg(Oct[2,1])
	# plotImg(Oct[2,2])
	# plotImg(Oct[2,3])

	# print Oct[1:,:-1].shape
	return Oct[1:,:-1]


def SS_supp_Intensity(OSMatrix):
	"""
	
	Create supplimenting layers for the intensity channel (every Octave and every scale!!) .
	Input: Octvave-Scale matrix
	Output: i, i^2 Matrices for every scale and every octave in the same format as the OSMatrix

	"""
	supInt1 = np.empty((OSMatrix.shape), dtype = 'object')
	supInt2 = np.empty((OSMatrix.shape), dtype = 'object')

	for i in range(OSMatrix.shape[0]):
		for j in range(OSMatrix.shape[1]):
			supInt1[i,j] = OSMatrix[i,j][:,:,0] 			# Taking only the intensity channel
			supInt2[i,j] = OSMatrix[i,j][:,:,0]**2.0 		# Take the square of all elements of the intensity channel

	return supInt1, supInt2

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

	print c1_bar_c[0,0]
	print c1_bar_c[0,1]

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

	#print sig_c_col.shape
	#print sig_c_col[0,0].shape
	#print sig_c_col[0,0][0,0]

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


def csNDIntensity(image):
	print "convert colorspace..."
	l1 = convertColorspace(image) 

	#print l1
	print "create DOG pyramid..."
	L,c1,c2 = getDOGPyramid(l1, level=5, sigmaX=10.5,sigmaY=10.5,ksize=(31,31))

	print L

	print "create supplimenting layers for intensity..."
	iSup = []
	for i in range(len(L)):
		intensitySup = supplementing_layers_intensity(L[i])
		iSup.append(intensitySup)


	# print (iSup)

	# plotImg(iSup[1][1])
	icSup = copy.deepcopy(iSup)
	isSup = copy.deepcopy(iSup)

	print "get center gaussian for all supplimenting layers of intensity..."
	for i in range(len(iSup)):
		for j in range(len(iSup[i])):
			icSup[i][j] = csEstimate(iSup[i][j], 1.0)

	#print icSup[0][0]#, iSup[0][0][0,0], isSup[0][0][0,0]

	#print iSup[0][0].shape
	print "center normal distribution for all intensity layers..."	
	# plotImg(iSup[1][1])
	cND_Int_mu = []
	for i in range(len(icSup)):
		cND_Int_mu.append(icSup[i][0])

	cND_Int_sigma = []
	for i in range(len(icSup)):
		cND_Int_sigma.append(icSup[i][1] - np.multiply(icSup[i][0], icSup[i][0]))

	print cND_Int_mu[0].shape

	print "get surround gaussian for all supplimenting layers of intensity..."
	for i in range(len(iSup)):
		for j in range(len(iSup[i])):
			isSup[i][j] = csEstimate(iSup[i][j], 10.0)

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


def csNDColor(image):
	print "convert colorspace..."
	l1 = convertColorspace(image) 

	print "create DOG pyramid..."
	L,c1,c2 = getDOGPyramid(l1, level=5, sigmaX=10.5,sigmaY=10.5,ksize=(31,31))


	print "create supplimenting layers for intensity..."
	iSup = []
	for i in range(len(L)):
		intensitySup = (supplementing_layers_color(c1[i], c2[i]))
		iSup.append(intensitySup)


	print len(iSup)

	#plotImg(iSup[1][1])
	icSup = copy.deepcopy(iSup)
	isSup = copy.deepcopy(iSup)

	print "get center gaussian for all supplimenting layers of color..."
	for i in range(len(iSup)):
		for j in range(len(iSup[i])):
			icSup[i][j] = (csEstimate(iSup[i][j], 1.0))
			#print iSup[i][j]

	print icSup[0][0][0,0], iSup[0][0][0,0], isSup[0][0][0,0]

	#print iSup[0][1].shape
	print "center normal distribution for all color layers..."	
	#plotImg(iSup[4][4])
	cND_Int_mu = []

	for i in range(len(icSup)):
		#print i
		line = np.empty(icSup[i][0].shape, dtype = 'object')
		for j in range(line.shape[0]):
			for k in range(line.shape[1]):
				#print (j,k)
				temparr = np.zeros((2,1))#, dtype = 'object')
				temparr[0,0] = icSup[i][0][j,k]
				temparr[1,0] = icSup[i][1][j,k]
				line[j,k] = temparr#.astype(np.double)

		cND_Int_mu.append(line)


	#print cND_Int_mu[0].shape

	cND_Int_sigma = []
	for i in range(len(icSup)):
		#print i
		line = np.empty(icSup[i][0].shape, dtype = 'object')
		for j in range(line.shape[0]):
			for k in range(line.shape[1]):
				#print (j,k)
				temparr = np.zeros((2,2))
				temparr[0,0] = icSup[i][2][j,k] - (icSup[i][0][j,k] ** 2)
				temparr[0,1] = icSup[i][4][j,k] - (icSup[i][0][j,k] * icSup[i][1][j,k])
				temparr[1,0] = icSup[i][4][j,k] - (icSup[i][0][j,k] * icSup[i][1][j,k])
				temparr[1,1] = icSup[i][3][j,k] - (icSup[i][1][j,k] ** 2)
				line[j,k] = temparr#.astype(np.double)
		cND_Int_sigma.append(line)


	print (cND_Int_sigma[0].shape)

	print "get center gaussian for all supplimenting layers of color..."
	for i in range(len(iSup)):
		for j in range(len(iSup[i])):
			isSup[i][j] = (csEstimate(iSup[i][j], 10.0))

	print "center normal distribution for all color layers..."	
	#plotImg(iSup[0][4])
	sND_Int_mu = []
	for i in range(len(isSup)):
		line = np.empty(isSup[i][0].shape, dtype = 'object')
		for j in range(line.shape[0]):
			for k in range(line.shape[1]):
				temparr = np.zeros((2,1))#, dtype = 'object')
				temparr[0,0] = isSup[i][0][j,k]
				temparr[1,0] = isSup[i][1][j,k]
				line[j,k] = temparr#.astype(np.double)

		sND_Int_mu.append(line)


	#print cND_Int_mu[1].shape

	sND_Int_sigma = []
	for i in range(len(isSup)):
		line = np.empty(isSup[i][0].shape, dtype = 'object')
		for j in range(line.shape[0]):
			for k in range(line.shape[1]):
				temparr = np.zeros((2,2))#, dtype = 'object')
				temparr[0,0] = isSup[i][2][j,k] - (isSup[i][0][j,k] ** 2)
				temparr[0,1] = isSup[i][4][j,k] - (isSup[i][0][j,k] * isSup[i][1][j,k])
				temparr[1,0] = isSup[i][4][j,k] - (isSup[i][0][j,k] * isSup[i][1][j,k])
				temparr[1,1] = isSup[i][3][j,k] - (isSup[i][1][j,k] ** 2)
				line[j,k] = temparr#.astype(np.double)

		#line = line.astype(np.double)
		sND_Int_sigma.append(line)



	return cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma


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
					t3 = 2.0 * np.sqrt(np.sqrt(cND_Int_sigma[i,j][p,q]) * sND_Int_sigma[i,j][p,q] * np.sqrt(cND_Int_sigma[i,j][p,q]))
					if (t1 + t2 - t3) < 0:
						tempimg[p,q] = 0.0
					else:
						tempimg[p,q] = np.sqrt(t1 + t2 - t3)

			WInt[i,j] = tempimg

	return WInt


def computeCSWassersteinIntensity(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma):
	WInt = []

	for i in range(len(cND_Int_mu)):
		print i
		tempimg = np.zeros((cND_Int_mu[i].shape))
		for j in range(cND_Int_mu[i].shape[0]):
			for k in range(cND_Int_mu[i].shape[1]):
				t1 = np.linalg.norm(cND_Int_mu[i][j,k]-sND_Int_mu[i][j,k])
				t1 = t1 * t1
				t2 = sND_Int_sigma[i][j,k] + cND_Int_sigma[i][j,k] 
				t3 = 2.0 * np.sqrt(np.sqrt(cND_Int_sigma[i][j,k]) * sND_Int_sigma[i][j,k] * np.sqrt(cND_Int_sigma[i][j,k]))

				if (t1 + t2 - t3) < 0:
					tempimg[j,k] = 0.0
				else:
					tempimg[j,k] = np.sqrt(t1 + t2 - t3)

		WInt.append(tempimg)


	return WInt

def matmult (A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
      print "Cannot multiply the two matrices. Incorrect dimensions."
      return

    # Create the result matrix
    # Dimensions would be rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]
    #print C

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k]*B[k][j]
    return C


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


def computeCSWassersteinColor(cND_Int_mu, cND_Int_sigma, sND_Int_mu, sND_Int_sigma):
	WInt = []
	#print (cND_Int_mu[0].shape), (cND_Int_sigma[0].shape), (sND_Int_mu[0].shape), (sND_Int_sigma[0].shape)
	for i in range(len(cND_Int_mu)):
		print i
		#print "shapeeeeeee", cND_Int_mu[0][0].shape
		tempimg = np.zeros((cND_Int_mu[i].shape))
		for j in xrange(cND_Int_mu[i].shape[0]):
			for k in xrange(cND_Int_mu[i].shape[1]):
				#print (i,j,k)
				t1 = np.linalg.norm(cND_Int_mu[i][j,k]-sND_Int_mu[i][j,k])
				#print t1
				t1 = t1 ** 2.0
				#print t1
				t2 = np.trace(sND_Int_sigma[i][j,k]) + np.trace(cND_Int_sigma[i][j,k]) 
				p1 = np.trace(np.dot(cND_Int_sigma[i][j,k], sND_Int_sigma[i][j,k]))
				p2 =  np.sqrt(abs((np.linalg.det(np.dot(cND_Int_sigma[i][j,k], sND_Int_sigma[i][j,k])))))
				t3 = 2.0 * np.sqrt(p1 + 2.0*p2)
				#print t3
				if (t1 + t2 - t3) < 0:
					tempimg[j,k] = 0.0
				else:
					tempimg[j,k] = np.sqrt(t1 + t2 - t3)
				#print tempimg[j,k]

		WInt.append(tempimg)


	return WInt

def combineScales(imglist):
	s = len(imglist)

	for i in range(s):
		if i == 0:
			pass
		else:
			imglist[i] = cv2.resize(imglist[i], (imglist[0].shape[1], imglist[0].shape[0]), interpolation=cv2.INTER_CUBIC)

	img = 0
	for i in range(s):
		img += (np.sqrt(i+1) * imglist[i])

	img = img/s
	return img

if __name__ == '__main__':
	image = readConvert('../testimages/dscn4311.jpg')
	OSMatrix = scaleSpaceRepresentation(image, scales = 3, octaves = 2)
	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Color(OSMatrix, 1.0, 10.0)

	print (mu_c_int[0,0][1,1], mu_s_int[0,0][1,1])
	print (mu_c_int[0,0][2,2], mu_s_int[0,0][2,2])

	WInt = SScomputeCSWassersteinColor(mu_c_int, sig_c_int, mu_s_int, sig_s_int)
	print WInt.shape
	print WInt[0,0] 

	print WInt[0,1]

	plotImg(WInt[0,0])
	plotImg(WInt[0,1])
	plotImg(WInt[0,2])
	plotImg(WInt[1,0])
	plotImg(WInt[1,1])
	plotImg(WInt[1,2])
	#print mu_c_int[0,0][0,0], sig_c_int[0,0][0,0]
	
	#plot1DND(mu_c_int[0,0][0,1], sig_c_int[0,0][0,1])
	