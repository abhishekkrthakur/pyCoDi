import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import cv2
from ssUtils import *

fname = '../f2.png'

def getimg():
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

#    plotImg2(image1)    
#    plotImg2(image2)

    t = 1.0
    im = (1-t) * image1  +  t * image2

    return im


neighborhood_size = 10
threshold = 70

data = getimg() #scipy.misc.imread(fname)

data_max = filters.maximum_filter(data, neighborhood_size)
maxima = (data == data_max)
data_min = filters.minimum_filter(data, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

#print diff

labeled, num_objects = ndimage.label(maxima)
print np.unique(labeled)#.shape
slices = ndimage.find_objects(labeled)
x, y = [], []
#print slices
for dy,dx, dz in slices:
    #print dz
    x_center = (dx.start + dx.stop - 1)/2
    x.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2    
    y.append(y_center)



imgFile = '/Users/abhishek/Documents/Thesis/ittis-images/coke/DBtraining/19.png'
print "converting image...."
image1 = readImg(imgFile)
data = cv2.resize(image1, (400, 300), interpolation=cv2.INTER_CUBIC)
data = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

print len(x)
labels = range(1,len(x)+1)
plt.imshow(data)
for label, x, y in zip(labels, x, y):
    print label, x, y
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.5))#,
        #arrowprops = dict(arrowstyle = 'fancy', connectionstyle = 'arc3,rad=0.3', shrinkB=5, color = '0.5'))

plt.show()
# plt.imshow(data)
# plt.savefig('../data.png', bbox_inches = 'tight')

# plt.autoscale(False)
# plt.plot(x,y, 'ro')
# plt.savefig('../result.png', bbox_inches = 'tight')