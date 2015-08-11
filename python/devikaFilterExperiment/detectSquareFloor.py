""" Here we try to detect the floor in which the mouse lives. The main assumption is that
the floor is square. We apply Otsu threshold + open + dilate + coutours + contour
approximation. The filtering techinique is the same as in rat finder. However, since here
we want to segment the opposite, there is a point in which we invert the image"""
import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from jasf import jasf_cv
from devika import devika_cv
from copy import deepcopy
from cv2 import imshow


cam = cv2.VideoCapture('../../video/avi/myFavoriteVideo.avi')

window_input = jasf_cv.getNewWindow('input')
window_output = jasf_cv.getNewWindow('output')
window_settings = jasf_cv.getNewWindow('settings')

def doNothing(opt):
    pass

cv2.createTrackbar('epsilon', window_settings, 34, 100, doNothing) 
cv2.createTrackbar('jump', window_settings, 50, 400, doNothing) 

cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

def detectFloor(input, alpha, previous_approx, previous_roi, allowed_jump):
    #leave Otsu decide the threshold
    ret, otsu_threshold = cv2.threshold(input, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #opening operation to fill holes and eliminate noise
    open = cv2.morphologyEx(otsu_threshold, cv2.MORPH_OPEN, cleaning_kernel)
    open = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

    #invert image
    ret, invert = cv2.threshold(open, 0.5, 1, cv2.THRESH_BINARY_INV)
    if len(previous_roi) > 0:
        invert = cv2.drawContours(invert, [previous_roi], 0, 1, -1)
    #find countours
    ret, cnts, ret2 = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #find approximations to degree determined by epsilon
    approx = [cv2.approxPolyDP(c, alpha*cv2.arcLength(c,True), True) for c in cnts] 
    #the floor is the one with the highest value of area
    roi = approx[np.argmax([cv2.contourArea(c) for c in approx])]

    prev_size = len(previous_roi)
    if (len(previous_roi) > 0) and ((len(roi) not in range(prev_size-2, prev_size+2+1)) or\
            (np.abs(cv2.contourArea(roi) - cv2.contourArea(previous_roi)) > allowed_jump)):
        return previous_roi, previous_approx, invert

    return roi, approx, invert

previous_approx = []
previous_roi = []
roi = []
approx = []
while cam.isOpened():
    ret, frame = cam.read()
    left, right = devika_cv.break_left_right(frame)
    B = right[:,:,0]
    input = B.copy()

    alpha = cv2.getTrackbarPos('epsilon', window_settings)/1000.0
    jump = cv2.getTrackbarPos('jump', window_settings)

    previous_approx = deepcopy(approx)
    previous_roi = deepcopy(roi)

    roi, approx, invert = detectFloor(input, alpha, previous_approx, previous_roi, jump)

    approxImage = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(approxImage, approx, -1, (0,0,200), 2)
    cv2.drawContours(approxImage, [roi], 0, (200,0,0), 3)

    cv2.imshow(window_input,255*invert)
    cv2.imshow(window_output, approxImage)

    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
        print 'finishing due to user command'
        break


cam.release()
cv2.destroyAllWindows()



