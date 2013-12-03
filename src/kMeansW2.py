"""
This script contains the functions used to calculate kMeans clustering
on a given set of 1-D and 2-D distributions in space.

The distance function used to calculate the centroids has been modified to be 
Wasserstein distance on Euclidean norm and thus preserves the original mean of 
the kMeans clustering algorithm.

Mean is not always the average and is most of the time not trivial 
(http://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf  see table 8.2 page 501)

Generalization isn't always compatible with optimized version

__author__ : Abhishek Thakur

"""
from __future__ import division
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
from ssUtils import *
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.cluster.vq import vq
import numpy as np
from ImageOperations import *
import pylab as pl
import copy
import matplotlib.cm as cm
import random
import sys
from time import time
from pylab import plot,show


def distanceFunction2D(X,Y):
	return w2distance2D(X[0], Y[0], X[1], Y[1])

def distanceFunction1D(X,Y):
	dist = np.zeros((X.shape[0], Y.shape[0]))
	#print X.shape, Y.shape
	for i in range(X.shape[0]):
		for j in range(Y.shape[0]):
			#print w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1])
			dist[i,j] = (w2distance1D(X[i,0], X[i,1], Y[j,0], Y[j,1]))

	return dist



def kmeans1D( X, centres, delta=.001, maxiter=10, p=2, verbose=0 ):
    """ centres, Xtocentre, distances = kmeans1D( X, initial centres ... )
    in:
        X N x dim  may be sparse
        centres k x dim: initial centres, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centres
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centrevec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centres, k x dim
        Xtocentre: each X -> its nearest centre, ints N -> k
        distances, N
    see also: kmeans1Dsample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centres = centres.todense() if issparse(centres) \
        else centres.copy()
    N, dim = X.shape
    k, cdim = centres.shape
    print X.shape
    if dim != cdim:
        raise ValueError( "kmeans1D: X %s and centres %s must have the same number of columns" % (
            X.shape, centres.shape ))
    if verbose:
        print "kmeans1D: X %s  centres %s  delta=%.2g  maxiter=%d  " % (
            X.shape, centres.shape, delta, maxiter)
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = distanceFunction1D(X, centres) #cdist_sparse( X, centres, metric=metric, p=p )
        #print D.shape, X.shape, centres.shape  # |X| x |centres|
        xtoc = D.argmin(axis=1)  # X -> nearest centre
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print "kmeans1D: av |X - nearest centre| = %.4g" % avdist
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centres[jc] = X[c].mean( axis=0 )
    if verbose:
        print "kmeans1D: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc)
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print "kmeans1D: cluster 50 % radius", r50.astype(int)
        print "kmeans1D: cluster 90 % radius", r90.astype(int)
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centres, xtoc, distances

def kmeans1Dsample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans1D, fast for large N:
        1) kmeans1D a random sample of nsample ~ sqrt(N) from X
        2) full kmeans1D, starting from those centres
    """
        # merge w kmeans1D ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centres = randomsample( X, int(k) )
    samplecentres = kmeans1D( Xsample, pass1centres, **kwargs )[0]
    return kmeans1D( X, samplecentres, **kwargs )


def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    

    return d

def randomsample( X, n ):
    """ 
    
    random.sample of the rows of X
    X may be sparse -- best csr

    """
    sampleix = random.sample( xrange( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcentres( X, centres, p=2 ):
    """ each X -> nearest centre, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centres, p=p )  # |X| x |centres|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()


# class Kmeans:
#     """ km = Kmeans( X, k= or centres=, ... )
#         in: either initial centres= for kmeans1D
#             or k= [nsample=] for kmeans1Dsample
#         out: km.centres, km.Xtocentre, km.distances
#         iterator:
#             for jcentre, J in km:
#                 clustercentre = centres[jcentre]
#                 J indexes e.g. X[J], classes[J]
#     """
#     def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
#         self.X = X
#         if centres is None:
#             self.centres, self.Xtocentre, self.distances = kmeans1Dsample(
#                 X, k=k, nsample=nsample, **kwargs )
#         else:
#             self.centres, self.Xtocentre, self.distances = kmeans1D(
#                 X, centres, **kwargs )

#     def __iter__(self):
#         for jc in range(len(self.centres)):
#             yield jc, (self.Xtocentre == jc)


if __name__ == "__main__":
	import random
	import sys
	from time import time


	print "load image..."
	image = readConvert('/Users/abhishek/Documents/Thesis/pyCoDi/pyCoDi/testimages/crop.jpg')
	OSMatrix = scaleSpaceRepresentation(image, scales = 2, octaves = 5)
	mu_c_int, sig_c_int, mu_s_int, sig_s_int = SSCS_Dist_Intensity(OSMatrix, 1.0, 10.0)
	#WInt1 = SScomputeCSWassersteinIntensity(mu_c_int, sig_c_int, mu_s_int, sig_s_int)

	#print cND_Int_mu
	mu_sigma = np.asarray(zip(mu_c_int[0,0].ravel(), sig_c_int[0,0].ravel())) #splitcenterdata(cND_Int_mu, cND_Int_sigma)
	data = mu_sigma

	X = data
	ncluster = 5
	kmdelta = .001
	kmiter = 100

	randomcentres = randomsample( X, ncluster )


	centres, xtoc, dist = kmeans1D( X, randomcentres,
		delta=kmdelta, maxiter=kmiter, verbose=2)


	print centres

	print xtoc

	idx,_ = vq(data,centres)

	centroids = centres

	plot(data[idx==0,0],data[idx==0,1],'ob',
	     data[idx==1,0],data[idx==1,1],'or',
	     data[idx==2,0],data[idx==2,1],'og',
	     data[idx==3,0],data[idx==3,1],'oy',
	     data[idx==4,0],data[idx==4,1],'oc')

	plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
	show()

    # N = 10000
    # dim = 2
    # ncluster = 10
    # kmsample = 100  # 0: random centres, > 0: kmeans1Dsample
    # kmdelta = .001
    # kmiter = 10
    # metric = "cityblock"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
    # seed = 1

    # exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    # np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    # np.random.seed(seed)
    # random.seed(seed)

    # print "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
    #     N, dim, ncluster, kmsample, metric)
    # X = np.random.exponential( size=(N,dim) )
    #     # cf scikits-learn datasets/
    # t0 = time()
    # if kmsample > 0:
    #     centres, xtoc, dist = kmeans1Dsample( X, ncluster, nsample=kmsample,
    #         delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    # else:
    #     randomcentres = randomsample( X, ncluster )
    #     centres, xtoc, dist = kmeans1D( X, randomcentres,
    #         delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    # print "%.0f msec" % ((time() - t0) * 1000)