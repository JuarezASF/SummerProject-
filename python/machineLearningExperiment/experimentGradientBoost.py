import cv2
import sklearn
from sklearn import cross_validation
from sklearn import svm
import numpy as np
import experimentUtil as util
import pickle

from sklearn.ensemble import GradientBoostingClassifier

print 'trying classifier with cross validation'
clf = GradientBoostingClassifier(n_estimators = 90, max_depth=4, learning_rate =0.05)
X,y = util.tryClassifier(clf, './data.pickle')

print 'training classifier on the entire data set'
clf.fit(X,y)

fn = './clf_gradientBoosting.pickle'
print 'saving classifier to', fn
with open(fn, 'w') as f:
    pickle.dump(clf, f)
