# Random forest classification instead of using the k-Means clustering algorithm

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
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics,cross_validation
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
	# print "Loading Image ///// Parameter adjustment is not allowed at the moment ///"

	# trainingfolder = sys.argv[1]
	# #testimage = sys.argv[2]

	# trainingimages = glob.glob(trainingfolder + '*.ppm')
	# #print trainingimages
	# positivesamples = []
	# negativesamples = []

	# print "lets create positive samples first"

	# for i in range(len(trainingimages)):
	# 	imgFile = trainingimages[i]
	# 	image = readConvert(imgFile)
	# 	screen, px = setup(imgFile)
	# 	left, upper, right, lower = mainLoop(screen, px)
	# 	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 3)
	# 	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)
	# 	mu_c_col, sig_c_col, mu_s_col, sig_s_col = SSCS_Dist_Color(OSMatrix, 1.0, 10.0)
	# 	mu_c_int_test, sig_c_int_test, mu_s_int_test, sig_s_int_test = cropTest(mu_c_int, sig_c_int, mu_s_int, sig_s_int, 
	#       																		left, upper, right, lower)
	# 	mu_c_col_test, sig_c_col_test, mu_s_col_test, sig_s_col_test = cropTest(mu_c_col, sig_c_col, mu_s_col, sig_s_col, 
	# 						     												left, upper, right, lower)
	# 	centroidsInt = RFInt(mu_c_int_test, sig_c_int_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)
	# 	centroidsCol = RFCol(mu_c_col_test, sig_c_col_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)


	# 	pos = np.hstack((centroidsInt, centroidsCol))#.tolist()
		
	# 	#print len(list(pos))
	# 	for j in range(pos.shape[0]):
	# 		positivesamples.append(pos[j,:])


	# print len(positivesamples)

	# print "time to create negative samples"

	# for i in range(len(trainingimages)):
	# 	imgFile = trainingimages[i]
	# 	image = readConvert(imgFile)
	# 	screen, px = setup(imgFile)
	# 	left, upper, right, lower = mainLoop(screen, px)
	# 	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 3)
	# 	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)
	# 	mu_c_col, sig_c_col, mu_s_col, sig_s_col = SSCS_Dist_Color(OSMatrix, 1.0, 10.0)
	# 	mu_c_int_test, sig_c_int_test, mu_s_int_test, sig_s_int_test = cropTest(mu_c_int, sig_c_int, mu_s_int, sig_s_int, 
	#       																		left, upper, right, lower)
	# 	mu_c_col_test, sig_c_col_test, mu_s_col_test, sig_s_col_test = cropTest(mu_c_col, sig_c_col, mu_s_col, sig_s_col, 
	# 						     												left, upper, right, lower)
	# 	centroidsInt = RFInt(mu_c_int_test, sig_c_int_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)
	# 	centroidsCol = RFCol(mu_c_col_test, sig_c_col_test, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)


	# 	neg = np.hstack((centroidsInt, centroidsCol))#.tolist()

	# 	for j in range(neg.shape[0]):
	# 		negativesamples.append(neg[j,:])


	# poslabels = [1] * len(positivesamples)
	# neglabels = [0] * len(negativesamples)


	# traindata = np.asarray(positivesamples + negativesamples)
	# labels = np.asarray(poslabels + neglabels)

	# cPickle.dump(traindata, open('../traindata3.pkl', 'wb'), -1)
	# cPickle.dump(labels, open('../labels3.pkl', 'wb'), -1)


	traindata = cPickle.load(open('../traindata3.pkl'))
	labels = cPickle.load(open('../labels3.pkl'))

	print traindata
	print labels


	clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=1, 
								min_samples_leaf=1, max_features=None, bootstrap=True, oob_score=True, n_jobs=-1, 
								random_state=None, verbose=2, min_density=None, compute_importances=None)


	#print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(clf, traindata, labels, cv=20, scoring='roc_auc', 
	#																											verbose = 2))



	print "loading test image"

	testfile = '/Users/abhishek/Documents/Thesis/images-td-exp-diss-simone/name-plate/dscn3712.ppm'

	print "converting image...."
	testimage = readConvert(testfile)

	print "creating OSMatrix"
	OSMatrixTest = scaleSpaceRepresentation(testimage, scales = 2, octaves = 3)

	print "processing for intensity channel"
	mu_c_intT, sig_c_intT, mu_s_intT, sig_s_intT = SSCS_Dist_Intensity(OSMatrixTest, 1.0, 10.0)

	print "processing for color channel"
	mu_c_colT, sig_c_colT, mu_s_colT, sig_s_colT = SSCS_Dist_Color(OSMatrixTest, 1.0, 10.0)

	centroidsInt = RFInt(mu_c_intT, sig_c_intT, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)
	centroidsCol = RFCol(mu_c_colT, sig_c_colT, n_iter = 1000, n_clusters = 3, delta = 0.001, verbose = 2)

	tst = np.hstack((centroidsInt, centroidsCol))#.tolist()

	WInt1 = SScomputeCSWassersteinIntensity(mu_c_intT, sig_c_intT, mu_s_intT, sig_s_intT)
	WInt2 = SScomputeCSWassersteinColor(mu_c_colT, sig_c_colT, mu_s_colT, sig_s_colT)

	tempmat1 = (SScombineScales(WInt1))
	tempmat2 = (SScombineScales(WInt2))
	t2 = (tempmat1 + tempmat2)/2.0

	testdata = []
	for j in range(tst.shape[0]):
		testdata.append(tst[j,:])

	clf.fit(traindata, labels)
	preds = clf.predict_proba(testdata)[:,1]

	for i in range(len(preds)):
		if preds[i] > 0.99:
			preds[i] = 1.0
		else:
			preds[i] = 0.0

	WInt = SScombineScales(convertRFPredToImg(preds, OSMatrixTest))
	WInt = smoothImg(WInt, 3,3,(9,9))

	#WInt = (t2 + WInt)/2.0
	plotImg(WInt)

	#testdata 




