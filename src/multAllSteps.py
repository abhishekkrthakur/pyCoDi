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
import pygame, sys
from PIL import Image
import cPickle

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

pygame.init()


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

def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
    	x += width
    	width = abs(width)
    if height < 0:
    	y += height
    	height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
    	return current
    if current == prior:
    	return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)

def setup(path):
	px = pygame.image.load(path)
	screen = pygame.display.set_mode( px.get_rect()[2:] )
	screen.blit(px, px.get_rect())
	pygame.display.flip()
	return screen, px

def mainLoop(screen, px):
	topleft = bottomright = prior = None
	n=0
	while n!=1:
		for event in pygame.event.get():
			if event.type == pygame.MOUSEBUTTONUP:
				if not topleft:
					topleft = event.pos
				else:
					bottomright = event.pos
					n=1
			if topleft:
				prior = displayImage(screen, px, topleft, prior)
	return ( topleft + bottomright )


# def displayImage( screen, px, topleft):
#     screen.blit(px, px.get_rect())
#     if topleft:
#         pygame.draw.rect( screen, 0, pygame.Rect(topleft[0], topleft[1], pygame.mouse.get_pos()[0] - topleft[0], pygame.mouse.get_pos()[1] - topleft[1]))
#     pygame.display.flip()

# def setup(path):
#     px = pygame.image.load(path)
#     screen = pygame.display.set_mode( px.get_rect()[2:] )
#     screen.blit(px, px.get_rect())
#     pygame.display.flip()
#     return screen, px

# def mainLoop(screen, px):
#     topleft = None
#     bottomright = None
#     runProgram = True
#     while runProgram:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 runProgram = False
#             elif event.type == pygame.MOUSEBUTTONUP:
#                 if not topleft:
#                     topleft = event.pos
#                 else:
#                     bottomright = event.pos
#                     runProgram = False
#         displayImage(screen, px, topleft)
#     return ( topleft + bottomright )



def locateImg(imfile, intImg, colImg):

	imgFile = intImg

	print "converting image...."
	image1 = readImg(imgFile)
	image1 = cv2.resize(image1, (400, 300), interpolation=cv2.INTER_CUBIC)
	print "Loading Image ///// Parameter adjustment is not allowed at the moment ///"

	imgFile = colImg

	print "converting image...."
	image2 = readImg(imgFile)
	image2 = cv2.resize(image2, (400, 300), interpolation=cv2.INTER_CUBIC)

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

	plotImg(im)	


	im = rgb2gray(im)
	#plotImg(im)	
	print im.shape
	image = im
	
	thresh = 150

	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			if im[i,j] > thresh:
				im[i,j] = 1
			else:
				im[i,j] = 0

	plotImg(im)

	imgFile = imfile

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
	plotImg(im)
	print im

	print "converting image...."
	image3 = readImg(imgFile)
	image3 = cv2.resize(image3, (400, 300), interpolation=cv2.INTER_CUBIC)	

	for i in range(image3.shape[2]):
		image3[:,:,i] = (image3[:,:,i] + imx[:,:,i])# + image3[:,:,i]


	savePlot(image3, '../f1.jpg')



def getTestimg(testfile, centroidsInt1, centroidsCol1, wtI1, wtC1):
	#testfile = '/Users/abhishek/Documents/Thesis/ittis-images/coke/DBtraining/43.png'

	print "converting image...."
	testimage = readConvert(testfile)
	#testimage = cv2.resize(testimage, (400, 300), interpolation=cv2.INTER_CUBIC)

	print "creating OSMatrix"
	OSMatrixTest = scaleSpaceRepresentation(testimage, scales = 2, octaves = 2)

	print "processing for intensity channel"
	mu_c_intT, sig_c_intT, mu_s_intT, sig_s_intT = SSCS_Dist_Intensity(OSMatrixTest, 1.0, 10.0)

	print "processing for color channel"
	mu_c_colT, sig_c_colT, mu_s_colT, sig_s_colT = SSCS_Dist_Color(OSMatrixTest, 1.0, 10.0)

	print "computeW2CentroidDiffInt"
	tempmat1 = computeW2CentroidDiffInt(centroidsInt1, wtI1, mu_c_intT, sig_c_intT)

	# cPickle.dump(centroidsCol1, open('../centroids.pkl', 'wb'), -1)
	# cPickle.dump(wtC1, open('../weights.pkl', 'wb'), -1)
	# cPickle.dump(mu_c_colT, open('../OSMatrixTestmu.pkl', 'wb'), -1)
	# cPickle.dump(sig_c_colT, open('../OSMatrixTestsigma.pkl', 'wb'), -1)

	print "computeW2CentroidDiffCol"
	tempmat2 = computeW2CentroidDiffCol(centroidsCol1, wtC1, mu_c_colT, sig_c_colT)


	return tempmat1, tempmat2

if __name__ == '__main__':
	print "Loading Image ///// Parameter adjustment is not allowed at the moment ///"

	imgFile = '/Users/abhishek/Documents/Thesis/images-td-exp-diss-simone/hl/b/1.png'

	print "converting image...."
	print (sys.argv[0])
	print (sys.argv[1])
	image = readConvert(imgFile)
	#image = cv2.resize(image, (400, 300), interpolation=cv2.INTER_CUBIC)

	# plotImg(image[:,:,0])
	# plotImg(image[:,:,1])
	# plotImg(image[:,:,2])


	print "ROI selection for test image"
	screen, px = setup(imgFile)
	#print screen
	#print px
	left, upper, right, lower = mainLoop(screen, px)
	#pygame.display.quit()

	print "creating OSMatrix"
	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 2)

	# plotImg(OSMatrix[0,0][:,:,0])
	# plotImg(OSMatrix[0,1][:,:,0])
	# plotImg(OSMatrix[1,0][:,:,0])
	# plotImg(OSMatrix[1,1][:,:,0])
	# plotImg(OSMatrix[2,0][:,:,0])
	# plotImg(OSMatrix[2,1][:,:,0])


	# plotImg(OSMatrix[0,0][:,:,1])
	# plotImg(OSMatrix[0,1][:,:,1])
	# plotImg(OSMatrix[1,0][:,:,1])
	# plotImg(OSMatrix[1,1][:,:,1])
	# plotImg(OSMatrix[2,0][:,:,1])
	# plotImg(OSMatrix[2,1][:,:,1])


	# plotImg(OSMatrix[0,0][:,:,2])
	# plotImg(OSMatrix[0,1][:,:,2])
	# plotImg(OSMatrix[1,0][:,:,2])
	# plotImg(OSMatrix[1,1][:,:,2])
	# plotImg(OSMatrix[2,0][:,:,2])
	# plotImg(OSMatrix[2,1][:,:,2])

	#cPickle.dump(OSMatrix, open('../OSMatrix_SCHREIBTISCH_DUNKEL2_0011_Scales_3___Octaves_3.pkl', 'wb'), -1)
	#print "pickle dumped"

	print "processing for intensity channel"
	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)

	print "processing for color channel"
	mu_c_col, sig_c_col, mu_s_col, sig_s_col = SSCS_Dist_Color(OSMatrix, 1.0, 10.0)

	print "extracting intensity regions from training image"
	mu_c_int_test, sig_c_int_test, mu_s_int_test, sig_s_int_test = cropTest(mu_c_int, sig_c_int, mu_s_int, sig_s_int, 
																			left, upper, right, lower)

	print "extracting color region from training image"
	mu_c_col_test, sig_c_col_test, mu_s_col_test, sig_s_col_test = cropTest(mu_c_col, sig_c_col, mu_s_col, sig_s_col, 
																			left, upper, right, lower)

	print mu_c_int[0,0], sig_c_int[0,0]

	print "clustering all scales and octaves of the test region - intensity"
	centroidsInt1, wtI1 = kMeansInt(mu_c_int_test, sig_c_int_test, n_iter = 50, n_clusters = 2, delta = 0.1, verbose = 2)

	print "clustering all scales and octaves of the test region - color"
	centroidsCol1, wtC1 = kMeansCol(mu_c_col_test, sig_c_col_test, n_iter = 50, n_clusters = 2, delta = 0.1, verbose = 2)




	for i in range(len(sys.argv)):
		#print sys.argv[i]
		tempmat1, tempmat2 = getTestimg(sys.argv[i+1], centroidsInt1, centroidsCol1, wtI1, wtC1)
		tempmat1 = (SScombineScales(tempmat1))
		tempmat2 = (SScombineScales(tempmat2))
		t = (tempmat1 + tempmat2)/2.0

		#plotImg(tempmat1)
		#plotImg(tempmat2)

		savePlot(tempmat1, '../temp1.png')
		savePlot(tempmat2, '../temp2.png')
		#plotImg(t)

		locateImg(sys.argv[i+1], '../temp1.png', '../temp2.png')

		print "####################### : :::: : ", i+1






	# plotImg(tempmat1[0,0])
	# plotImg(tempmat1[0,1])
	# plotImg(tempmat1[1,0])
	# plotImg(tempmat1[1,1])
	# plotImg(tempmat1[2,0])
	# plotImg(tempmat1[2,1])

	# plotImg(tempmat2[0,0])
	# plotImg(tempmat2[0,1])
	# plotImg(tempmat2[1,0])
	# plotImg(tempmat2[1,1])
	# plotImg(tempmat2[2,0])
	# plotImg(tempmat2[2,1])

	# tempmat1 = (SScombineScales(tempmat1))
	# tempmat2 = (SScombineScales(tempmat2))
	# t = (tempmat1 + tempmat2)/2.0

	# plotImg(tempmat1)
	# plotImg(tempmat2)

	# savePlot(tempmat1, '../temp1.png')
	# savePlot(tempmat2, '../temp2.png')
	# plotImg(t)

	# locateImg(testfile, '../temp1.png', '../temp2.png')




