"""
Image Operations

Translation of CoDi saliency code from C++ to Python

"""
import numpy as np
import math
import cv2
import pylab as pl
import matplotlib.cm as cm

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

	# Create the Intensity channel...

	for j in range(outImg.shape[0]):
		for i in range(outImg.shape[1]): 
			outImg[j,i,0] = (src[j,i,0] + src[j,i,1] + src[j,i,2]) / 3.0
			outImg[j,i,1] = src[j,i,0] - src[j,i,1]
			outImg[j,i,2] = src[j,i,2] - (src[j,i,0] + src[j,i,1]) / 2.0

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
    lapimg = sm - gpyr #gpyr-sm
    return lapimg

def readImg(filename):
	"""
	Read image using the opencv imread function and return result as a 
	numpy array which is RGB and not BGR
	"""
	img = cv2.imread(filename)
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



#load and show an image in gray scale
image = readImg('../testimages/popout_color_redgreen1.jpg')

print image.shape
print "converting image"
l1 = convertColorspace(image) #(image[:,:,0], level = 1)
i,c1,c2 = getDOGPyramid(l1, level=5, sigmaX=1.5,sigmaY=1.0,ksize=(5,5))
#i = createLaplacianPyramid(l1[:,:,0],sigmaX=1.5,sigmaY=1.0,ksize=(5,5), level=1)
# print l1.shape
print c1[0].shape, c1[1].shape, c1[2].shape
#print i.shape
#cv2.imshow('image', i[0])
#cv2.waitKey(0)
pl.imshow(i[0], cmap = cm.gray)  # @UndefinedVariable
pl.show()

pl.imshow(i[1], cmap = cm.gray)  # @UndefinedVariable
pl.show()

pl.imshow(i[2], cmap = cm.gray)  # @UndefinedVariable
pl.show()

pl.imshow(i[3], cmap = cm.gray)  # @UndefinedVariable
pl.show()

pl.imshow(i[4], cmap = cm.gray)  # @UndefinedVariable
pl.show()
# #             
#             
    

# def gaussianBoxBlur(src, dst, sigmaX, sigmaY, n, borderType, filterPrecision):
# 	if (sigmaY <= 0.0):
# 		sigmaY = sigmaX

# 	"""
# 	cv::Mat src = _src.getMat();
# 	_dst.create(src.size(), src.type());
# 	cv::Mat dst = _dst.getMat();
# 	"""

# 	wx = 1
# 	wy = 1
# 	mx = n
# 	my = n

# 	sX = sigmaX
# 	sY = sigmaY

# 	ksizeW, ksizeH = max(2 * (int) (filterPrecision * sX) + 1, 3), max(2 * (int) (filterPrecision * sY) + 1, 3)

# 	wx = (((((int) (math.sqrt((12.0 * sigmaX * sigmaX) / n + 1.0))) + 1) / 2) * 2) - 1

# 	mx = math.ceil(((12 * sigmaX * sigmaX) - (n * wx * wx + 4 * n * wx + 3 * n)) / (-4 * wx - 4))

# 	sX = sqrt(sigmaX * sigmaX - ((-mx * (4 * wx + 4) + n * wx * wx + 4 * n * wx + 3 * n) / 12.0))

# 	if (n < (ksizeW - max(2 * (int) (filterPrecision * sX) + 1, 3))):
# 		ksizeW = max(2 * (int) (filterPrecision * sX) + 1, 3)
# 	else:
# 		wx = 1
# 		mx = n

# 	wy = (((((int) (math.sqrt((12.0 * sigmaY * sigmaY) / n + 1.0))) + 1) / 2) * 2) - 1
# 	my = math.ceil(((12 * sigmaY * sigmaY) - (n * wy * wy + 4 * n * wy + 3 * n)) / (-4 * wy - 4))

# 	sY = sqrt(sigmaY * sigmaY - ((-my * (4 * wy + 4) + n * wy * wy + 4 * n * wy + 3 * n) / 12.0))

# 	if (n < (ksizeH - max(2 * (int) (filterPrecision * sY) + 1, 3))):
# 		ksizeH = max(2 * (int) (filterPrecision * sY) + 1, 3)
# 	else:
# 		wy = 1
# 		my = n

# 	if (wx > 1 or wy > 1):
# 		sumType = cv2.CV_MAKETYPE(cv2.CV_64F, src.channels())
# 		rowFilter = getRowSumFilter(src.type(), sumType, wx)
# 		columnFilter = getColumnSumFilter(sumType, dst.type(), wy, -1, 1. / (wx * wy))





















