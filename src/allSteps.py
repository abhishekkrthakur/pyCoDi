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






if __name__ == '__main__':
	print "Loading Image ///// Parameter adjustment is not allowed at the moment ///"

	imgFile = '/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/pix2.png'

	print "converting image...."
	image = readConvert(imgFile)

	print "ROI selection for test image"
	screen, px = setup(imgFile)
	left, upper, right, lower = mainLoop(screen, px)
	#pygame.display.quit()

	print "creating OSMatrix"
	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 4)

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
	centroidsInt1, wtI1 = kMeansInt(mu_c_int_test, sig_c_int_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)

	print "clustering all scales and octaves of the test region - color"
	centroidsCol1, wtC1 = kMeansCol(mu_c_col_test, sig_c_col_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)




	# imgFile = '/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/art3.jpg'

	# print "converting image...."
	# image = readConvert(imgFile)

	# print "ROI selection for test image"
	# screen, px = setup(imgFile)
	# left, upper, right, lower = mainLoop(screen, px)

	# print "creating OSMatrix"
	# OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 3)

	# cPickle.dump(OSMatrix, open('../OSMatrix_SCHREIBTISCH_DUNKEL2_0011_Scales_3___Octaves_3.pkl', 'wb'), -1)
	# print "pickle dumped"

	# print "processing for intensity channel"
	# mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)

	# print "processing for color channel"
	# mu_c_col, sig_c_col, mu_s_col, sig_s_col = SSCS_Dist_Color(OSMatrix, 1.0, 10.0)

	# print "extracting intensity regions from training image"
	# mu_c_int_test, sig_c_int_test, mu_s_int_test, sig_s_int_test = cropTest(mu_c_int, sig_c_int, mu_s_int, sig_s_int, 
	# 																		left, upper, right, lower)

	# print "extracting color region from training image"
	# mu_c_col_test, sig_c_col_test, mu_s_col_test, sig_s_col_test = cropTest(mu_c_col, sig_c_col, mu_s_col, sig_s_col, 
	# 																		left, upper, right, lower)

	# print "clustering all scales and octaves of the test region - intensity"
	# centroidsInt2 = kMeansInt(mu_c_int_test, sig_c_int_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)

	# print "clustering all scales and octaves of the test region - color"
	# centroidsCol2 = kMeansCol(mu_c_col_test, sig_c_col_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)


	# centroidsInt = np.vstack((centroidsInt1, centroidsInt2))
	# centroidsCol = np.vstack((centroidsCol1, centroidsCol2))


	testfile = '/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/pix2.png'

	print "converting image...."
	testimage = readConvert(testfile)

	print "creating OSMatrix"
	OSMatrixTest = scaleSpaceRepresentation(testimage, scales = 2, octaves = 4)

	print "processing for intensity channel"
	mu_c_intT, sig_c_intT, mu_s_intT, sig_s_intT = SSCS_Dist_Intensity(OSMatrixTest, 1.0, 10.0)

	print "processing for color channel"
	mu_c_colT, sig_c_colT, mu_s_colT, sig_s_colT = SSCS_Dist_Color(OSMatrixTest, 1.0, 10.0)

	print "computeW2CentroidDiffInt"
	tempmat1 = computeW2CentroidDiffInt(centroidsInt1, wtI1, mu_c_intT, sig_c_intT)

	print "computeW2CentroidDiffCol"
	tempmat2 = computeW2CentroidDiffCol(centroidsCol1, wtC1, mu_c_colT, sig_c_colT)

	#WInt1 = SScomputeCSWassersteinIntensity(mu_c_intT, sig_c_intT, mu_s_intT, sig_s_intT)
	#WInt2 = SScomputeCSWassersteinColor(mu_c_colT, sig_c_colT, mu_s_colT, sig_s_colT)

	tempmat1 = (SScombineScales(tempmat1))
	tempmat2 = (SScombineScales(tempmat2))
	t = (tempmat1 + tempmat2)/2.0

	plotImg(tempmat1)
	plotImg(tempmat2)
	plotImg(t)





