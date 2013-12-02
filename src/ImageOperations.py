"""
Image Operations

Translation of CoDi saliency code from C++ to Python
__author__ : Abhishek Thakur

"""
import numpy as np
#import math
import cv2
import pylab as pl
import matplotlib.cm as cm
import time
import math

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

def convertColorspace(src):
	"""
	Convert the colorspace of RGB Image to a much psychologically
	motivated colorspace. Given by Simone Frintrop and Dominik Klein in 
	CoDi saliency.
	
	Input : RGB Image (Grayscale image will throw an error) | Image[R;G;B]

	Output: Converted Image with Three channels. One Intensity and two Color channels.

	"""

	sizeImg = src.shape[2]

	if not (sizeImg == 3):
		raise AssertionError("Only RGB Input is accepted at the moment...")

	outImg = np.zeros((src.shape))

	outImg[:,:,0] = (src[:,:,0] + src[:,:,1] + src[:,:,2]) / 3.0
	outImg[:,:,1] = src[:,:,0] - src[:,:,1]
	outImg[:,:,2] = src[:,:,2] - (src[:,:,0] + src[:,:,1]) / 2.0

	# Create the Intensity channel...

	# for j in range(outImg.shape[0]):
	# 	for i in range(outImg.shape[1]): 
	# 		outImg[j,i,0] = (src[j,i,0] + src[j,i,1] + src[j,i,2]) / 3.0
	# 		outImg[j,i,1] = src[j,i,0] - src[j,i,1]
	# 		outImg[j,i,2] = src[j,i,2] - (src[j,i,0] + src[j,i,1]) / 2.0

	return outImg


def getDOGPyramid(image, level, sigmaX=1.5,sigmaY=1.0,ksize=(5,5)):
	"""
	inputs:
		image : must be a three dimensional array with the new colorspace
		sigmaX : sigma in X direction, default value is 1.5
		sigmaY : sigma in Y direction, default value is 1.0
		ksize : size of the gaussian kernel
		level : the level of pyramid that is required

	output: three lists consisting of all the required pyramid levels for 
			i, c1 and c2. 

			Note: i[0] will give the lowest level.
	"""

	intList  = []
	c1List = []
	c2List = []

	lvl = level

	for i in range(lvl):
		intList.append(np.asarray(createGaussianPyramid(image[:,:,0], sigmaX=sigmaX, sigmaY=sigmaY, ksize=ksize, level = i)))
		c1List.append(np.asarray(createGaussianPyramid(image[:,:,1], sigmaX=sigmaX, sigmaY=sigmaY, ksize=ksize, level = i)))
		c2List.append(np.asarray(createGaussianPyramid(image[:,:,2], sigmaX=sigmaX, sigmaY=sigmaY, ksize=ksize, level = i)))

	return intList, c1List, c2List


def getDOGPyramid_OLD(image, level, sigmaX=1.5,sigmaY=1.0,ksize=(5,5)):
	"""
	inputs:
		image : must be a three dimensional array with the new colorspace
		sigmaX : sigma in X direction, default value is 1.5
		sigmaY : sigma in Y direction, default value is 1.0
		ksize : size of the gaussian kernel
		level : the level of pyramid that is required

	output: three lists consisting of all the required pyramid levels for 
			i, c1 and c2. 

			Note: i[0] will give the lowest level.
	"""

	intList  = []
	c1List = []
	c2List = []

	lvl = level

	for i in range(lvl):
		intList.append(np.asarray(createLaplacianPyramid(image[:,:,0], sigmaX=sigmaX, sigmaY=sigmaY, ksize=ksize, level = i)))
		c1List.append(np.asarray(createLaplacianPyramid(image[:,:,1], sigmaX=sigmaX, sigmaY=sigmaY, ksize=ksize, level = i)))
		c2List.append(np.asarray(createLaplacianPyramid(image[:,:,2], sigmaX=sigmaX, sigmaY=sigmaY, ksize=ksize, level = i)))

	return intList, c1List, c2List



def smoothImg(image, sigmaX, sigmaY, ksize):
    """
    OpenCV implementation of the Gaussian filter. 
    Filter is applied to each dimension individually if an RGB image is passed
    """
    smoothed = cv2.GaussianBlur(image,ksize,sigmaX, sigmaY)
    return smoothed



def pyr_lap(image,sigmaX=1.5,sigmaY=1.0,ksize=(5,5), level=1):
    '''
    Returns a given level of Gaussian Pyramid.
    '''
    currImg, i = image, 0
    while i < level:
        smooth = smoothImg(currImg, sigmaX,sigmaY,ksize)
        currImg = smooth[::2, ::2]
        final_img = smooth
        i += 1
    
    return final_img

def createGaussianPyramid(image,sigmaX=1.5,sigmaY=1.0,ksize=(5,5), level=1):
    '''
    Returns a given level of Gaussian Pyramid.
    '''
    currImg, i = image, 0
    while i < level:
        smooth = smoothImg(currImg, sigmaX,sigmaY,ksize)
        smooth = smooth[::2, ::2]
        currImg = smooth
        i += 1
    
    return currImg


def createLaplacianPyramid(image,sigmaX=1.5,sigmaY=1.0,ksize=(5,5), level=1):
    '''
    Returns a given level of Laplacian Pyramid.
    The Laplacian Pyramid has been approximated by using the Difference of Gaussians (DoG)
    '''
    gpyr =  createGaussianPyramid(image,sigmaX,sigmaY,ksize, level) 
    sm = pyr_lap(image,sigmaX,sigmaY,ksize, level+1)
    lapimg = gpyr-sm
    return lapimg


def readImg(filename):
	"""
	Read image using the opencv imread function and return result as a 
	numpy array which is RGB and not BGR
	"""
	img = cv2.imread(filename,cv2.CV_LOAD_IMAGE_COLOR)
	image = np.zeros((img.shape)) 
	image[:,:,0] = img[:,:,2]
	image[:,:,1] = img[:,:,1]
	image[:,:,2] = img[:,:,0]

	return image


def readConvert(filename):
	"""
	Read image and convert the colorspace...
	"""
	img = cv2.imread(filename)
	image = np.zeros((img.shape)) 
	image[:,:,0] = img[:,:,2]
	image[:,:,1] = img[:,:,1]
	image[:,:,2] = img[:,:,0]
	image = convertColorspace(image)

	return image

@timing
def supplementing_layers_intensity(img):
	"""
	This function adds supplementing layers to the image. 
	Input: A layer of the DoG pyramid 
	Output: Supplementing layers for the Intensity. The output has a list containing all the layers.
			Output can be seen as L : [i, i**2]
	"""

	sizeImg = len(img.shape)

	if not (sizeImg == 2):
		raise AssertionError("Input to the supplementing_layers_intensity() function should be an image with 1 channels!")


	L = []

	i = img
	i2 = np.multiply(i,i)
	L.append(i)
	L.append(i2)

	return L

@timing
def supplementing_layers_color(img1,img2):
	"""
	This function adds supplementing layers to the image. 
	Input: A layer of the DoG pyramid 
	Output: Supplementing layers for the Color. The output has a list containing all the layers.
			Output can be seen as L : [c1,c2,c1**2,c2**2,c1*c2]
	"""

	sizeImg = len(img1.shape)

	if not (sizeImg == 2):
		raise AssertionError("Input to the supplementing_layers_intensity() function should be an image with 1 channels!")


	L = []

	c1 = (img1)
	c2 = img2
	c1_2 = c1**2.0 #np.multiply(c1,2.0)
	c2_2 = c2 **2.0#np.multiply(c2,2.0)
	print c1.shape, c2.shape
	c1c2 = c1 * c2 #np.dot(c1,c2)

	L.append(c1)
	L.append(c2)
	L.append(c1_2)
	L.append(c2_2)
	L.append(c1c2)

	return L

def csEstimate(image, std):
	"""
	Estimation of the center __mu__ and __sigma__ for the 
	Normal Distributions of Intensity channel

	Input: Image and standard deviation

	"""

	ks = int(math.ceil(3*std))
	#print ks
	if ks%2 == 0:
		ks += 1
	#print ks
	ksize = (ks,ks)
	#print image
	return smoothImg(image, std, std, ksize)






if __name__ == '__main__':
	#load and show an image
	image = readImg('../testimages/dscn4311.jpg')

	pl.imshow(image, cmap = cm.gray)  # @UndefinedVariable
	pl.show()

	#print image.shape
	#print "converting image"
	l1 = convertColorspace(image) #(image[:,:,0], level = 1)
	L,c1,c2 = getDOGPyramid(l1, level=5, sigmaX=1.2,sigmaY=1.0,ksize=(5,5))
	#i = createLaplacianPyramid(l1[:,:,0],sigmaX=1.5,sigmaY=1.0,ksize=(5,5), level=1)
	# print l1.shape
	#print c1[0].shape, c1[1].shape, c1[2].shape
	#print i.shape
	#cv2.imshow('image', i[0])
	#cv2.waitKey(0)

	#L = supplementing_layers_color(c1[4], c2[4])#(i[0])
	#L = supplementing_layers_intensity(i[4])#(i[0])
	#print L[0]
	#print L[1]

	pl.imshow(L[0], cmap = cm.gray)  # @UndefinedVariable
	pl.show()

	pl.imshow(L[1], cmap = cm.gray)  # @UndefinedVariable
	pl.show()

	pl.imshow(L[2], cmap = cm.gray)  # @UndefinedVariable
	pl.show()
	pl.imshow(L[3], cmap = cm.gray)  # @UndefinedVariable
	pl.show()
	pl.imshow(L[4], cmap = cm.gray)  # @UndefinedVariable
	pl.show()
