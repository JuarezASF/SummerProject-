import numpy as np
import cv2

cam = cv2.VideoCapture("../video/mp4/2014-07-16_08-41-11.mp4")
fgbg = cv2.BackgroundSubtractorMOG(history=20, nmixtures=10, backgroundRatio=0.7)

while (cam.isOpened()):
    ret, frame = cam.read()
    blured = cv2.blur(frame, ksize=(7,7))
    fgmask = fgbg.apply(blured)

    cv2.imshow('input',frame)
    cv2.imshow('output',fgmask)


    k = cv2.waitKey(2) & 0xff
    if k == ord('q'):
       break
cam.release()
cv2.destroyAllWindows()
