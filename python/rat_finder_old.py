"""This version is a translation to Python from Devika's Matlab code 'rat finder'. I am
not used to computer vision with matlab, so I decided to translate it """
import numpy as np
import cv2
from jasf import jasf_cv
from devika import devika_cv
from copy import deepcopy

def myFilter(contours, minMassTh, maxMassTh):
    o = []
    for c in contours:
        area = cv2.contourArea(c)/100
        if (area >=minMassTh) and (area <= maxMassTh):
            o.append(c)
    return o

def findRat(input, th, th_max):
    global cleaning_kernel
    #leave Otsu decide the threshold
    ret, otsu_threshold = cv2.threshold(input, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #opening operation to fill holes and eliminate noise
    open = cv2.morphologyEx(otsu_threshold, cv2.MORPH_OPEN, cleaning_kernel)
    open = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    #filter small objects
    filterSmall = jasf_cv.filterObjects(open.copy(), th, th_max)
    target = filterSmall.copy()
    output, contours, hier = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, otsu_threshold, open, filterSmall

def myDrawContours(output, cnts):
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, cnts, -1, (255,0,0), 2)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(output, (x,y), (x+w, y+h), (0,255,0), 2)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output, [box], 0, (0,0,255), 2)
        center = jasf_cv.getCenterOfContour(c) 
        cv2.circle(output, center, 2, (0,0,255), 2)
    return output

import Tkinter
import tkMessageBox
def initialize(img, th, th_max):
    cnts = findRat(img, th, th_max) 
    cnts = cnts[0]
    if len(cnts) == 1:
        #if there is only one initial center, return it as the rat
        rx,ry =jasf_cv.getCenterOfContour(cnts[0]) 
        return rx,ry, cnts[0]
    else:
        #if there are more, then we ask the user to help us
        global ratInitialized, mouse, rx, ry
        ratInitialized = False
        rx,ry = -1,-1
        tkMessageBox.showinfo("Alert!", "We need you to inform the initial position of the\
                rat! Please click on the rat on window 'initial position'")

        def getMousePosition(event, x, y, flags, param):
            """ mouse callback to set the rat position. This function gets the user press
            position and compare it with the known centers, picking the closest match"""
            global rx, ry, ratInitialized,mouse
            if event == cv2.EVENT_LBUTTONUP:
                print 'clicked on', x, y
                centers = [jasf_cv.getCenterOfContour(c) for c in cnts]
                print len(centers)
                distances = [np.linalg.norm(np.array(c) - np.array((x,y))) for c in centers]
                i = np.argmin(distances)
                rx,ry = centers[i]
                mouse = cnts[i] 
                if pointDistance((rx,ry), (x,y)) > 20:
                    print 'didnt work!'
                    tkMessageBox.showinfo("Alert!", "The position clicked is not close\
                            enough to one of my centers!")
                else: 
                    print 'mouse set'
                    ratInitialized = True

        cv2.namedWindow('initial position', cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('initial position', getMousePosition)

        while ratInitialized == False:
            #read thresholds
            th = cv2.getTrackbarPos('th', window_settings)
            th_max = cv2.getTrackbarPos('max', window_settings)
            #find contours
            cnts = findRat(img, th, th_max) 
            cnts = cnts[0]
            #draw
            img2show = myDrawContours(img, cnts)
            #show
            cv2.imshow('initial position', img2show)
            cv2.waitKey(5)
        cv2.destroyWindow('initial position')

        return rx, ry, mouse
def pointDistance(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

class MouseNotFound(Exception):
    def __init__(self, reason = ''):
        self.reason = reason

def filterMouse(cnts, last_center):
    """ Receives a set of countours and the last known position of the mouse and use a set
    of heuristics to choose the rat """
    rx,ry,mouse = -1,-1,[]
    try:
        if len(cnts) == 0:
            raise MouseNotFound('Detection sees nothing!')
        px,py = last_center
        centers = [jasf_cv.getCenterOfContour(c) for c in cnts]
        distances = [pointDistance(np.array(c) , np.array((px,py))) for c in centers]
        i = np.argmin(distances)
        rx,ry = centers[i]
        mouse = cnts[i] 

        if pointDistance((rx,ry), (px,py)) > 80:
            print 'new center', rx,ry
            print 'previous center', px,py
            raise MouseNotFound('Closest center not close enough to previous position! Is this rat a ninja or what??')

    except MouseNotFound as E:
        print 'rat not found!'
        print E.reason
        return last_center[0], last_center[1], False

    return rx,ry,mouse


#initialize camera
cam = cv2.VideoCapture('../video/avi/myFavoriteVideo.avi')
#the problem with this one is that the rat merges with the back panel
#cam = cv2.VideoCapture('../video/avi/2014-07-16_10-41-14.avi')

window_input = jasf_cv.getNewWindow('input')
window_otsu = jasf_cv.getNewWindow('otsu')
window_open = jasf_cv.getNewWindow('open')
window_small_filter = jasf_cv.getNewWindow('smallFilter')
window_output = jasf_cv.getNewWindow('output')
window_settings = jasf_cv.getNewWindow('settings')

cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

def doNothing(val):
    pass

cv2.createTrackbar('th', window_settings, 0, 400, doNothing)
cv2.createTrackbar('max', window_settings, 100, 400, doNothing)
cv2.createTrackbar('area_delta', window_settings, 4, 10, doNothing)


rx,ry,mouse = -1, -1,[]
while cam.isOpened():
    ret, frame = cam.read()
    if frame == None:
        print 'finishing due to end of video'
        break
    left, right = devika_cv.break_left_right(frame)

    th = cv2.getTrackbarPos('th', window_settings)
    th_max = cv2.getTrackbarPos('max', window_settings)
    area_delta = cv2.getTrackbarPos('area_delta', window_settings)

    hsv = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)

    B,G,R = cv2.split(right)
    H,S,V = cv2.split(hsv)

    input = B

    if (rx,ry) == (-1, -1):
        rx,ry,mouse = initialize(input, th, th_max)
        area = cv2.contourArea(mouse)/100
        newTh = max(area - area_delta, 9)
        newTh_max = max(area + area_delta, 16)
        cv2.setTrackbarPos('th', window_settings, int(newTh))
        cv2.setTrackbarPos('max', window_settings, int(newTh_max))

    contours, otsu_threshold, open, filterSmall = findRat(input, th, th_max)

    rx,ry,new_mouse = filterMouse(contours, (rx, ry))
    if type(new_mouse) is not bool:
        mouse = new_mouse
        area = cv2.contourArea(mouse)/100
        newTh = max(area - area_delta, 9)
        newTh_max = max(area + area_delta, 16)
        cv2.setTrackbarPos('th', window_settings, int(newTh))
        cv2.setTrackbarPos('max', window_settings, int(newTh_max))

    
    output = input.copy()

    output = myDrawContours(output, [mouse])
    
    cv2.imshow(window_otsu, 255*otsu_threshold)
    cv2.imshow(window_open, 255*open)
    cv2.imshow(window_small_filter, 255*filterSmall)
    cv2.imshow(window_input, input) 
    cv2.imshow(window_output, output) 

    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        print 'end of execution due to user command'
        break

cam.release()
cv2.destroyAllWindows()

