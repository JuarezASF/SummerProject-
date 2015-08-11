import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

#define examples for class learning
mean = [(0,0), (5,5)]
cov = [scp.diag([1,1]), scp.diag([0.5,1.2])]

N = 4000

dataA = np.array([np.random.multivariate_normal(mean[0], cov[0]) for i in range(N)])
dataB = np.array([np.random.multivariate_normal(mean[1], cov[1]) for i in range(N)])

#create data set for classification
ds = ClassificationDataSet(2,1,nb_classes = 2)
for x in dataA:
    ds.addSample(x, [0])
for x in dataB:
    ds.addSample(x, [1])

#split into training and test data
tstdata_temp, trndata_temp = ds.splitWithProportion(0.25)

#convert data sets back to ClassificationDataSet
#this is required due to a bug on the current version
tstdata = ClassificationDataSet(2, 1, nb_classes=2)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(2, 1, nb_classes=2)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

#convert to appropriate format for classification
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]

#build network
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

#build trainer
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

ticks = scp.arange(-10.0, 10.0, 0.2)
X, Y = scp.meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=2)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

for i in range(1):
    trainer.trainEpochs(1)
    
print 'error on training data', percentError( trainer.testOnClassData(), trndata['class'] )
print 'error on test data', percentError( trainer.testOnClassData( dataset=tstdata ), tstdata['class'] )


out = fnn.activateOnDataset(griddata)
out = out.argmax(axis=1)  # the highest output activation gives the class

print out.shape

X0, X1, Y0, Y1 = [],[],[],[]
for i in range(out.shape[0]):
    if out[i] == 0:
        X0.append(X.ravel()[i])
        Y0.append(Y.ravel()[i])
    if out[i] == 1:
        X1.append(X.ravel()[i])
        Y1.append(Y.ravel()[i])


plt.scatter(dataA[:,0], dataA[:,1], c = 'r')
plt.scatter(dataB[:,0], dataB[:,1], c = 'b')
plt.scatter(X0, Y0, c = 'r', alpha=0.5, marker = '.')
plt.scatter(X1, Y1, c = 'b', alpha=0.5, marker = '.')

plt.show()
