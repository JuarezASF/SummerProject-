import cv2
import sklearn
from sklearn import cross_validation
from sklearn import svm
import numpy as np
import experimentUtil as util
import pickle

from sklearn.ensemble import RandomForestClassifier

print 'trying classifier with cross validation'
clf = RandomForestClassifier(n_estimators = 90, max_depth=60, criterion='gini', n_jobs=-1)
X,y = util.tryClassifier(clf, './data.pickle')

print 'training classifier on the entire data set'
clf.fit(X,y)

fn = './clf_randomForest.pickle'
print 'saving classifier to', fn
with open(fn, 'w') as f:
    pickle.dump(clf, f)
