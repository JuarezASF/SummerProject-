import sys
sys.path.insert(0, '../')
import cv2
import jasf
from jasf import jasf_cv
from devika import devika_cv
from multiFiltersDataStructures import *
import pathlib
import pickle

def reportOnCollections(collectionList, action = 'loaded/saved'):
    print 'collection with', len(collectionList), 'filters', action
    for i,c in enumerate(collectionList):
        print 'filter', i
        print '\t', c.filterDescription
        print '\t # of row samples', len(c.rowSamples)
        print '\t # of col samples', len(c.colSamples)

def convertBackToClassificationDataSet(ds, nb):
    trndata = ClassificationDataSet(ds['input'][0].shape[1], 1, nb_classes=nb)
    for n in xrange(0, ds.getLength()):
        trndata.addSample( ds.getSample(n)[0], ds.getSample(n)[1] )

    return trndata
    
