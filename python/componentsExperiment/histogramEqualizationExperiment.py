import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from cv2 import imshow
from jasf import jasf_cv
from jasf import jasf_ratFinder
import jasf
from devika import devika_cv
import util

from config import video2load
cam = cv2.VideoCapture(video2load)

jasf.cv.getManyWindows(['G', 'B', 'R', 'H', 'S', 'V'], dim = (180,150), n=(6,6))
jasf.cv.getManyWindows(['Go', 'Bo', 'Ro', 'Ho', 'So', 'Vo'], dim = (180,150), n=(6,6))
jasf.cv.getManyWindows(['out'], dim = (180,150), n=(6,6))
jasf_cv.getSettingsWindow(n=(6,6))


control_filterType = 'free'
jasf_cv.setTrackbar('free/band pass', 0, 1)
jasf_cv.setTrackbar('l/c', 0, 255)
jasf_cv.setTrackbar('u/b', 100, 255)

def readSettings():
    global control_filterType
    control_filterType = 'free' if jasf.cv.readTrackbar('free/band pass') == 0 else 'band'
    lowerb, upperb = -1,-1
    if control_filterType == 'free':
        lowerb = jasf.cv.readTrackbar('l/c') 
        upperb = jasf.cv.readTrackbar('u/b') 
    else:
        center = jasf.cv.readTrackbar('l/c') 
        band = jasf.cv.readTrackbar('u/b') 

        lowerb = center - band
        upperb = center + band

    return lowerb, upperb

control_mode = 'run'

while cam.isOpened():
    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
     print 'finishing due to user input'
     break

    if ch == ord('p'):
        control_mode = 'pause' if control_mode == 'run' else 'run'

    if control_mode == 'run':
        ret,frame = cam.read()

    if ret == False:
        print 'finishing due to end of video'
        break

    lowerb, upperb = readSettings()

    B,G,R,H,S,V = inputList = util.getGBRHSVcomponents(frame.copy())
    inputList = jasf.cv.equalizeHist(inputList)
    inputList = jasf.cv.inRange(inputList, lowerb, upperb)

    Bo,Go,Ro,Ho,So,Vo = inputList 

    ret, bSo = jasf.cv.binarize(So)
    ret, ibSo = jasf.cv.invertBoolean(bSo)

    ret, bVo = jasf.cv.binarize(Vo)

    out = ibSo * bVo * 255


    jasf.cv.imshow(['G', 'B', 'R', 'H', 'S', 'V'], [B,G,R,H,S,V])
    jasf.cv.imshow(['Go', 'Bo', 'Ro', 'Ho', 'So', 'Vo'], [Bo,Go,Ro,Ho,So,Vo])
    cv2.imshow('out', out)




cv2.destroyAllWindows()
cam.release()
