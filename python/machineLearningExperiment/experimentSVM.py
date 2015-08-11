import cv2
import sklearn
from sklearn import cross_validation
from sklearn import svm
import numpy as np
import experimentUtil as util
import pickle

clf = svm.SVC()
util.tryClassifier(clf, './data.pickle')
