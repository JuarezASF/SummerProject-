"""Prorgam to play with the parameters of the Canny filter"""
import sys
sys.path.insert(0, '../')
import cv2
import numpy as np
from matplotlib import pyplot as plt
from jasf import jasf_cv
import jasf
import devika   

from config import video2load
cam = cv2.VideoCapture(video2load)

jasf_cv.getBasicWindows()
jasf.cv.setManyTrackbars(['th_min', 'th_max', 'th', 'n'], [100, 150, 100, 3], [255,255,255, 11])

while cam.isOpened():
    ret, frame = cam.read()
    if ret == False:
        print 'finishing due to end of video'
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    th_min, th_max, th, n= jasf.cv.readManyTrackbars(['th_min', 'th_max', 'th', 'n'])

    result = cv2.Canny(img,th_min,th_max, apertureSize=3, L2gradient=True) 

    result = cv2.pyrDown(result)

    ret, result = cv2.threshold(result, th, 255, cv2.THRESH_BINARY)
    if n>0:
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (n,n)))

    cv2.imshow('output',result)

    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
        print "execution being terminated due to press of key 'q'"
        break

cam.release()
cv2.destroyAllWindows()
