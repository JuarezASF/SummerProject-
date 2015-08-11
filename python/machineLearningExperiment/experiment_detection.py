import pickle
import sys
sys.path.insert(0, '../')
import cv2
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
import jasf
from jasf import jasf_cv
from devika import devika_cv
import experimentUtil as util

clf = pickle.load(open('./clf_randomForest.pickle', 'r'))

cam = cv2.VideoCapture('../../video/mp4/myFavoriteVideo.mp4')

jasf_cv.getBasicWindows()

while cam.isOpened():
    c = cv2.waitKey(5) & 0xFF
    if c == ord('q'):
        print 'finishing due to user input'
        break
        
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    down = cv2.pyrDown(gray)
    w,h = down.shape
    X = np.zeros_like(util.img2x(down[0:30, 0:30]))
    centers = []
    i_step, j_step = 10,10
    for i in range(15, w-15, i_step):
        for j in range(15, h-15, j_step):
            X = np.vstack((X,util.img2x(down[i-15:i+15, j-15:j+15])))
            centers.append((j,i))

    X = X[1:,:]
    y = clf.predict(X)
    toPaint = [c for i,c in enumerate(centers) if y[i] == 1]

    out = cv2.cvtColor(down, cv2.COLOR_GRAY2BGR)

    for c in toPaint:
        cv2.circle(out, c, 2, (255,0,0), -1)



    cv2.imshow('input', frame)
    cv2.imshow('output', out)

cv2.destroyAllWindows()
cam.release()
