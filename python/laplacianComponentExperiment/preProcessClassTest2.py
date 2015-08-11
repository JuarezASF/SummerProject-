help =  "\n\n HELP \n\nThis program will test the class FloorDetectPreProcess_S_based.  This class crops the original \
video file to select the area it believes is the ground texture. The ground texture is \
captured by the value of a pixel with its surrounding. The comparsion is made in the S component of the color space.\n\n"

print help import sys sys.path.insert(0, '../') import numpy as np import cv2 from cv2 import imshow from jasf import
jasf_cv from jasf import jasf_ratFinder from devika import devika_cv from util import *

cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves.mp4')
cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves2.mp4')
cam = cv2.VideoCapture('../../video/avi/myFavoriteVideo.avi')

filter = FloorDetectPreProcess_S_based()

jasf_cv.getNewWindow('input')
jasf_cv.getNewWindow('s')
jasf_cv.getNewWindow('output')
jasf_cv.getNewWindow('settings')

window_trackbar = 'settings'
#create track bars to stabilish color limits
def nothing(e):
    pass
cv2.createTrackbar('window_size',window_trackbar,5,13,nothing)
cv2.createTrackbar('th_min',window_trackbar,107,255,nothing)
cv2.createTrackbar('th_max',window_trackbar,193,255,nothing)
cv2.createTrackbar('erode',window_trackbar,3,13,nothing)
cv2.createTrackbar('dilate',window_trackbar,4,13,nothing)
cv2.createTrackbar('left/right',window_trackbar,0,1,nothing)


while cam.isOpened():
    ret,frame = cam.read()
    if ret == False:
        print 'finishing due to end of video'
        break
    left, right = devika_cv.break_left_right(frame)

    #read trackbraks
    window_size = max(3,cv2.getTrackbarPos('window_size', window_trackbar))
    th_min = max(1, cv2.getTrackbarPos('th_min', window_trackbar))
    th_max = max(1, cv2.getTrackbarPos('th_max', window_trackbar))
    erode_p = cv2.getTrackbarPos('erode', window_trackbar)
    dilate_p = cv2.getTrackbarPos('dilate', window_trackbar)
    control_left_right = cv2.getTrackbarPos('left/right', window_trackbar)

    input = left if control_left_right == 0 else right
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    filter.setParams(window_size, th_min, th_max, erode_p, dilate_p)

    s, mask = filter.preProcess(hsv[:222,:,1])
    output = gray[:222,:]*mask
    

    cv2.imshow('input', input)
    cv2.imshow('output', output)
    cv2.imshow('s', s)

    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
     print 'finishing due to user input'
     break

cv2.destroyAllWindows()
cam.release()


