#combine images

from skimage.filter import threshold_otsu, threshold_adaptive
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
from ssUtils import *
from PIL import Image
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction, dilation, disk
from skimage import filter
import copy


from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as pp



def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background

    return detected_peaks

def get_std(image):
    return np.std(image)

def get_max(image,sigma,alpha=20,size=10):
    i_out = []
    j_out = []
    image_temp = copy.deepcopy(image)
    while True:
        k = np.argmax(image_temp)
        j,i = np.unravel_index(k, image_temp.shape)
        if(image_temp[j,i] >= alpha*sigma):
            i_out.append(i)
            j_out.append(j)
            x = np.arange(i-size, i+size)
            y = np.arange(j-size, j+size)
            xv,yv = np.meshgrid(x,y)
            image_temp[yv.clip(0,image_temp.shape[0]-1),
                                   xv.clip(0,image_temp.shape[1]-1) ] = 0
            print xv
        else:
            break
    return i_out,j_out

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

if __name__ == '__main__':
	print "Loading Image ///// Parameter adjustment is not allowed at the moment ///"

	imgFile = '../temp1.png'

	print "converting image...."
	image1 = readImg(imgFile)
	image1 = cv2.resize(image1, (400, 300), interpolation=cv2.INTER_CUBIC)
	print "Loading Image ///// Parameter adjustment is not allowed at the moment ///"

	imgFile = '../temp2.png'

	print "converting image...."
	image2 = readImg(imgFile)
	image2 = cv2.resize(image2, (400, 300), interpolation=cv2.INTER_CUBIC)

	plotImg2(image1)	
	plotImg2(image2)

	# t = 0.1
	# im = (1-t) * image1  +  t * image2

	# plotImg(im)

	# t = 0.4
	# im = (1-t) * image1  +  t * image2

	# plotImg(im)

	# t = 0.6
	# im = (1-t) * image1  +  t * image2

	# plotImg(im)

	# t = 0.8
	# im = (1-t) * image1  +  t * image2

	# plotImg(im)

	# t = 0.9
	# im = (1-t) * image1  +  t * image2

	# plotImg(im)

	t = 0.5
	im = (1-t) * image1  +  t * image2
	savePlot(im, '../f2.jpg')



	im = rgb2gray(im)
	#plotImg(im)	
	print im.shape
	image = im



	thresh = 200

	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i,j] > thresh:
				im[i,j] = 1
			else:
				im[i,j] = 0


	detected_peaks = detect_peaks(image)
	print detected_peaks
#pp.subplot(4,2,(2*i+1))
#pp.imshow(paw)
#pp.subplot(4,2,(2*i+2) )
	pp.imshow(detected_peaks)

	pp.show()

#	plotImg2(im)

	imgFile = '/Users/abhishek/Documents/Thesis/ittis-images/coke/DBtraining/38.png'

	#plotImg(image3)
	im = filter.canny(im, sigma=1)
	selem = disk(1.5)
	im = dilation(im, selem)
	#im = invertImg(im)
	#im = im +500
	imx = np.zeros((im.shape[0], im.shape[1], 3))

	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i,j] > thresh:
				imx[i,j, 0] = 0
				imx[i,j, 1] = 150
				imx[i,j, 2] = 0
			else:
				imx[i,j, 0] = 0
				imx[i,j, 1] = 0
				imx[i,j, 2] = 0
	plotImg2(im)
	print im

	print "converting image...."
	image3 = readImg(imgFile)
	image3 = cv2.resize(image3, (400, 300), interpolation=cv2.INTER_CUBIC)	

	for i in range(image3.shape[2]):
		image3[:,:,i] = (image3[:,:,i] + imx[:,:,i])# + image3[:,:,i]


	savePlot(image3, '../f1.jpg')

