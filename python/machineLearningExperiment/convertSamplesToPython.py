import sys
sys.path.insert(0, '../')
import cv2
import numpy as np
import jasf
from jasf import jasf_cv
from devika import devika_cv
import pathlib
from experimentUtil import getData
import experimentUtil as util

positeExamplesFiles = list(p.absolute().as_posix() for p in pathlib.Path('./samples/positive/').iterdir() if p.suffix == '.png')
negativeExamplesFiles = list(p.absolute().as_posix() for p in pathlib.Path('./samples/negative/').iterdir() if p.suffix == '.png')

print 'number of positive files to be read:', len(positeExamplesFiles) 
print 'number of negatie files to be read:', len(negativeExamplesFiles) 


filter2BeUsed = util.pyrDownGray_filter
size = (30,30)
pX, missP = getData(positeExamplesFiles, expectedSize = size, filter = filter2BeUsed)
nX, missN = getData(negativeExamplesFiles, expectedSize = size, filter = filter2BeUsed)

print 'number of accepted positive files read', pX.shape[0]
print 'number of accepted negative files read', nX.shape[0]

print '# of images of inappropriate size', missP + missN

pY = np.ones((pX.shape[0], 1))
nY = np.zeros((nX.shape[0], 1))

X = np.vstack((pX, nX))
Y = np.vstack((pY, nY))


from experimentUtil import jDataSet
data = jDataSet(X,Y)

import pickle

fn = './data.pickle'
with open(fn, 'w') as f:
    pickle.dump([data], f)
