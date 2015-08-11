help =  "\n\n HELP \n\nThis program will test the class FloorDetectPreProcess. This class crops the original\
video file to select the area it believes is the ground texture. The ground texture is\
captured by using what I've been calling devika's filter(based on otsu threshould followed\
clianing operations. \n\n"

print help


import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from cv2 import imshow
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from util import *

cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves.mp4')
cam = cv2.VideoCapture('../../video/avi/myFavoriteVideo.avi')
cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves2.mp4')

filter = FloorDetectPreProcess()

jasf_cv.getNewWindow('input')
jasf_cv.getNewWindow('invert')
jasf_cv.getNewWindow('cnts')
jasf_cv.getNewWindow('output')

while cam.isOpened():
    ret,frame = cam.read()
    if ret == False:
        print 'finishing due to end of video'
        break
    left, right = devika_cv.break_left_right(frame)

    output, otsu_th, invert, cnts = filter.preProcess(right[:,:,0])
    
    cnts = cv2.drawContours(right.copy(), cnts, -1, (255,0,0), 2)

    cv2.imshow('input', right)
    cv2.imshow('invert', 255*invert)
    cv2.imshow('cnts', cnts)
    cv2.imshow('output', output)

    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
     print 'finishing due to user input'
     break

cv2.destroyAllWindows()
cam.release()

