"""Prorgam to play with the parameters of the Canny filter"""
import sys
sys.path.insert(0, '../')
import cv2
import numpy as np
from matplotlib import pyplot as plt
from jasf import jasf_cv

from config import video2load
cam = cv2.VideoCapture(video2load)

#cam = cv2.VideoCapture(0)
output = jasf_cv.getNewWindow('Canny Output')
track_window = jasf_cv.getNewWindow('Settings')

def onChangeCallBack():
    pass
cv2.createTrackbar('th_min', track_window, 0, 255, onChangeCallBack) 
cv2.setTrackbarPos('th_min', track_window, 100)

cv2.createTrackbar('th_max', track_window, 0, 255, onChangeCallBack) 
cv2.setTrackbarPos('th_max', track_window, 150)


while cam.isOpened():
    ret, frame = cam.read()
    #frame = cv2.blur(frame, (7,7))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    th_min = cv2.getTrackbarPos('th_min', track_window)
    th_max = cv2.getTrackbarPos('th_max', track_window)

    result = cv2.Canny(img,th_min,th_max, apertureSize=3, L2gradient=True) 

    cv2.imshow(output,result)

    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
        print "execution being terminated due to press of key 'q'"
        break

cam.release()
cv2.destroyAllWindows()

