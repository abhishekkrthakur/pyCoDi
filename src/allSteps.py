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
pygame.init()

def displayImage( screen, px, topleft):
    screen.blit(px, px.get_rect())
    if topleft:
        pygame.draw.rect( screen, 0, pygame.Rect(topleft[0], topleft[1], pygame.mouse.get_pos()[0] - topleft[0], pygame.mouse.get_pos()[1] - topleft[1]))
    pygame.display.flip()

def setup(path):
    px = pygame.image.load(path)
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px

def mainLoop(screen, px):
    topleft = None
    bottomright = None
    runProgram = True
    while runProgram:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                runProgram = False
            elif event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    runProgram = False
        displayImage(screen, px, topleft)
    return ( topleft + bottomright )






if __name__ == '__main__':
	print "Loading Image ///// Parameter adjustment is not allowed at the moment ///"

	imgFile = '/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/SCHREIBTISCH_DUNKEL2_0011.jpg'

	print "converting image...."
	image = readConvert(imgFile)

	print "ROI selection for test image"
	screen, px = setup(imgFile)
	left, upper, right, lower = mainLoop(screen, px)

	print "creating OSMatrix"
	OSMatrix = scaleSpaceRepresentation(image, scales = 3, octaves = 3)

	cPickle.dump(OSMatrix, open('../OSMatrix_SCHREIBTISCH_DUNKEL2_0011_Scales_3___Octaves_3.pkl', 'wb'), -1)
	print "pickle dumped"

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

	print "clustering all scales and octaves of the test region - intensity"
	centroidsInt = kMeansInt(mu_c_int_test, sig_c_int_test, n_iter = 100, n_clusters = 3, delta = 0.001, verbose = 0)

	print "clustering all scales and octaves of the test region - color"
	centroidsCol = kMeansCol(mu_c_col_test, sig_c_col_test, n_iter = 100, n_clusters = 3, delta = 0.001, verbose = 0)


	testfile = '/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/SCHREIBTISCH_DUNKEL2_0011.jpg'

	print "converting image...."
	testimage = readConvert(testfile)

	print "creating OSMatrix"
	OSMatrixTest = scaleSpaceRepresentation(testimage, scales = 3, octaves = 3)

	print "processing for intensity channel"
	mu_c_intT, sig_c_intT, mu_s_intT, sig_s_intT = SSCS_Dist_Intensity(OSMatrixTest, 1.0, 10.0)

	print "processing for color channel"
	mu_c_colT, sig_c_colT, mu_s_colT, sig_s_colT = SSCS_Dist_Color(OSMatrixTest, 1.0, 10.0)

	print "computeW2CentroidDiffInt"
	tempmat1 = computeW2CentroidDiffInt(centroidsInt, mu_c_intT, sig_c_intT)

	print "computeW2CentroidDiffCol"
	tempmat2 = computeW2CentroidDiffCol(centroidsCol, mu_c_colT, sig_c_colT)

	# WInt = SScomputeCSWassersteinIntensity(mu_c_intT, sig_c_intT, mu_s_intT, sig_s_intT)

	#WInt = SScombineScales(WInt)
	tempmat1 = (SScombineScales(tempmat1))
	tempmat2 = (SScombineScales(tempmat2))
	t = (tempmat1 + tempmat2)/2.0



	#plotImg(WInt)
	plotImg(tempmat1)
	plotImg(tempmat2)
	plotImg(t)
	# for i in range(tempmat1.shape[0]):
	# 	for j in range(tempmat1.shape[1]):
	# 		if tempmat1[i][j] == 0:
	# 			tempmat1[i][j] = 255
			#else:
			#	tempmat1[i][j] = 0
	#tempmat1 = tempmat1[tempmat1 == 0] = 255.0
	#plotImg(tempmat1)
	#plotImg(tempmat2)


	# print centroidsCol.shape
	# print centroidsCol[0,0]#.shape



	#print left, upper, right, lower
	#plotImg(image[upper:lower,left:right])

	# im = Image.open(imgFile)
	# im = im.crop(( left, upper, right, lower))
	# #print np.asarray(im)

	# OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 5)
	# mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)
	# # WInt1 = SScomputeCSWassersteinIntensity(mu_c_int, sig_c_int, mu_s_int, sig_s_int)
	# mu_c_col, sig_c_col, mu_s_col, sig_s_col = SSCS_Dist_Color(OSMatrix, 1.0, 10.0)
	# # WInt2 = SScomputeCSWassersteinColor(mu_c_col, sig_c_col, mu_s_col, sig_s_col)




