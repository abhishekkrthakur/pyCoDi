"""
Implement functions to calculate the Gaussian Pyramid.  

__author__ Abhishek Thakur

"""
import cv2
import numpy as np
import math 
import pylab as pl
import matplotlib.cm as cm

def smoothImg(image, sigmaX, sigmaY, ksize):
    """
    Gaussian filter. Filter is applied to each dimension individually
    if an RGB image is passed
    """
    smoothed = cv2.GaussianBlur(image,ksize,sigmaX, sigmaY)
    return smoothed
            
def pyr_lap(image,sigmaX=1.5,sigmaY=1.0,ksize=(5,5), level=1):
    '''
    Returns a given level of Gaussian Pyramid.
    According to VOCUS, the input image must be CIE-L*a*b
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
    According to VOCUS, the input image must be CIE-L*a*b
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
    According to VOCUS, the input image must be CIE-L*a*b
    '''
    gpyr =  createGaussianPyramid(image,sigmaX,sigmaY,ksize, level) 
    sm = pyr_lap(image,sigmaX,sigmaY,ksize, level+1)
    lapimg = gpyr-sm
    return lapimg
