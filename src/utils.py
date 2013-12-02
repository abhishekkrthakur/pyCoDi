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
	print Oct
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

	print Oct[1:,:-1].shape
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
	sig_c_col = np.empty((OSMatrix.shape), dtype = 'object')


	# Find c1_bar, c2_bar, c1_2_bar, c2_2_bar, c1c2_bar  |||||| for center and surround
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
			c1_bar_c[i,j] = smoothImg(c1[i,j], sizeIn)
			c1_bar_s[i,j] = smoothImg(c1[i,j], sizeOut)

			c1_2_bar_c[i,j] = smoothImg(c1_2[i,j], sizeIn)
			c1_2_bar_s[i,j] = smoothImg(c1_2[i,j], sizeOut)

			c2_2_bar_c[i,j] = smoothImg(c2_2[i,j], sizeIn)
			c2_2_bar_s[i,j] = smoothImg(c2_2[i,j], sizeOut)

			c1c2_bar_c[i,j] = smoothImg(c1c2[i,j], sizeIn)
			c1c2_bar_s[i,j] = smoothImg(c1c2[i,j], sizeOut)


	

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
	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)
	print mu_c_int[0,0][0,0], sig_c_int[0,0][0,0]
	plot1DND(mu_c_int[0,0][0,1], sig_c_int[0,0][0,1])
