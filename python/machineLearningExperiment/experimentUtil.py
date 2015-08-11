import cv2
import numpy as np
import pickle
import sklearn
from sklearn import cross_validation

def noneFilter(img):
    return img

def V_Filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[:,:,2]

def G_Filter(img):
    return img[:,:,0]

def Gray_Filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def pyrDownGray_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    down = cv2.pyrDown(gray)
    return down

def img2x(img):
    return img.reshape((1,img.size)) 


def getData(files, expectedSize = (60,60), filter = noneFilter):
    img = filter(cv2.imread(files[0]))
    X = img2x(img)
    missCount = 0
    for file in files[1:]:
        img = filter(cv2.imread(file))
        if img.shape[:2] == expectedSize:
            x = img2x(img)
            X = np.vstack((X,x))
        else:
            missCount += 1

    return X, missCount

class jDataSet:
    def __init__(self, X, Y):
        self.data = X
        self.target = Y

def tryClassifier(clf, dataFile):
    #=================
    #Load Data
    #=================
    print 'loading data...'
    loadData = pickle.load(open(dataFile, 'r'))
    data = loadData[0]
    X = data.data
    y = np.ravel(data.target)

    #=================
    #Evaluate Predictor
    #=================
    print 'evaluating predictor...'
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return X,y


