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

	imgFile = '/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/crop.jpg'

	print "converting image...."
	image = readConvert(imgFile)

	print "ROI selection for test image"
	screen, px = setup(imgFile)
	left, upper, right, lower = mainLoop(screen, px)
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




