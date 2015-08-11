import sys
sys.path.insert(0, '../')
import cv2
import sklearn
from jasf import jasf_cv
import jasf
from devika import devika_cv
import pathlib
import pickle

from pybrain.datasets import SupervisedDataSet 
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError 

from multiFiltersDataStructures import Sample
from multiFiltersDataStructures import FilterSampleCollection

from filterExperimentUtil import reportOnCollections
def convertSampleCollection2SupervisedDataSet(data):
    """ Receive on FilterSampleCollection as input and return two SupervisedDataSet objects. One for the col samples and
    the other for the row samples"""
    inputData = rowData, colData =  data.rowSamples, data.colSamples
    outputDataSet = []
    for D in inputData:
        dataSet  = ClassificationDataSet(len(D[0].data),1, nb_classes=2)
        for i,sample in enumerate(D):
            dataSet.addSample(sample.data, sample.target)

        dataSet._convertToOneOfMany()
        outputDataSet.append(dataSet)

    return outputDataSet


def getMultiplayerFeedForwardNetwork(inputLayerLen, hiddenLayersLenList, outLayerLen = 1):
    #create net
    net = FeedForwardNetwork()
    #create layers
    inLayer = LinearLayer(inputLayerLen, name='inLinearLayer')
    hiddenLayers = [SigmoidLayer(n, name='sigmoidLayer'+str(i)) for i,n in enumerate(hiddenLayersLenList)]
    outLayer = LinearLayer(outLayerLen, name='outLinearLayer')
    #add layers to net
    net.addInputModule(inLayer)
    for l in hiddenLayers:
        net.addModule(l)
    net.addOutputModule(outLayer)
    #create connections
    layers = [inLayer] + hiddenLayers + [outLayer]
    connections = [FullConnection(layers[i], layers[i+1], name='connection' + str(i)) for i in range(len(layers)-1)]
    #add connections to net
    for c in connections:
        net.addConnection(c)
    #do some required initialization
    net.sortModules()

    return net

def convert2ClassificationDataSet(data):
    modified = ClassificationDataSet(data['input'].shape[1], data['target'].shape[1], nb_classes=2)
    for n in xrange(0, data.getLength()):
        modified.addSample( data.getSample(n)[0], data.getSample(n)[1] )

    return modified


#main:
if __name__ == '__main__':
    #this gives a list of Collections, one for every filter type
    #every collection has rowSamples and colSamples
    data = pickle.load(open('./collectionList.pickle', 'r'))
    reportOnCollections(data, 'loaded')
    #this gives a list of (rowSamples, colSamples) where *Samples is a SupervisedDataSet
    data = [convertSampleCollection2SupervisedDataSet(d) for d in data]
    #now we split it into row and column data
    rowData = [d[0] for d in data]
    colData = [d[1] for d in data]

    print 'training row classifiers...'
    inputDataList = rowData
    clfs = []
    for d in inputDataList:
        train, test = d.splitWithProportion(0.9)
        dim = d['input'].shape[1]
        #convert data sets back to ClassificationDataSet
        #this is required due to a bug on the current version
        train = convert2ClassificationDataSet(train)
        test = convert2ClassificationDataSet(test)


        net = buildNetwork(dim, dim, dim/2, dim/2, 2)
        clfs.append(net)
        trainer = BackpropTrainer(net, train)
        print 'training classifier for 5 epochs...'
        for i in range(5):
            trainer.train()
        print 'error on training data', percentError( trainer.testOnClassData(), train['class'] )
        print 'error on test data', percentError( trainer.testOnClassData( dataset=test ), test['class'] )
        print np.max(net.activateOnDataset(train), axis = 1)
        quit()

        
