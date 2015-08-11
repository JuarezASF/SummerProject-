import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from jasf import jasf_cv

from config import video2load
cam = cv2.VideoCapture(video2load)

Bwindow = jasf_cv.getNewWindow('Blue Component')
Gwindow = jasf_cv.getNewWindow('Green component')
Rwindow = jasf_cv.getNewWindow('Red component')

Hwindow = jasf_cv.getNewWindow('Hue Component')
Swindow = jasf_cv.getNewWindow('Saturation component')
Vwindow = jasf_cv.getNewWindow('Value component')

jasf_cv.getNewWindow('canny')
jasf_cv.getNewWindow('settings')

jasf_cv.setTrackbar('canny_min')
jasf_cv.setTrackbar('canny_max')

while cam.isOpened():

    canny_min = cv2.getTrackbarPos('canny_min', 'settings')
    canny_max = cv2.getTrackbarPos('canny_max', 'settings')

    ret, frame = cam.read()
    B,G,R = cv2.split(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)

    canny = cv2.Canny(cv2.equalizeHist(V), canny_min, canny_max)

    cv2.imshow(Bwindow, B)
    cv2.imshow(Gwindow, G)
    cv2.imshow(Rwindow, R)

    cv2.imshow(Hwindow, H)
    cv2.imshow(Swindow, S)
    cv2.imshow(Vwindow, V)

    cv2.imshow('canny', canny)
    ch =  cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
        print "execution being terminated due to press of key 'q'"
        break

cam.release()
cv2.destroyAllWindows()

    
    

