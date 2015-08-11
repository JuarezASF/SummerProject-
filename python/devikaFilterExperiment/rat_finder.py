"""This version is a translation to Python from Devika's Matlab code 'rat finder'. I am
not used to computer vision with matlab, so I decided to translate it """
import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from copy import deepcopy

def myFilter(contours, minMassTh, maxMassTh):
    o = []
    for c in contours:
        area = cv2.contourArea(c)/100
        if (area >=minMassTh) and (area <= maxMassTh):
            o.append(c)
    return o

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
def initialize(img, th, th_max, otsu_th):
    """Find the countours on the first frame. If there are more than one, then we need
    help from the user to select which contour to track"""
    cnts = jasf_ratFinder.detectInterestingContours(img, th, th_max, otsu_th) 
    cnts = cnts[0]#we don't need the other results for this
    if len(cnts) == 1:
        #if there is only one initial center, return it as the rat
        rx,ry =jasf_cv.getCenterOfContour(cnts[0]) 
        return rx,ry, cnts[0]
    else:
        #if there are more, then we ask the user to help us
        global ratInitialized, mouse, rx, ry
        ratInitialized = False
        rx,ry = -1,-1
        tkMessageBox.showinfo("Alert!", "We need you to inform the initial position of the rat! Please click on the rat on window 'initial position'")

        def getMousePosition(event, x, y, flags, param):
            """ mouse callback to set the rat position. This function gets the user press
            position and compare it with the known centers, picking the closest match"""
            global rx, ry, ratInitialized,mouse
            if event == cv2.EVENT_LBUTTONUP:
                print 'clicked on', x, y
                #compute center of current contours and their distances to the user click
                centers = [jasf_cv.getCenterOfContour(c) for c in cnts]
                distances = [np.linalg.norm(np.array(c) - np.array((x,y))) for c in centers]
                #the mouse is the one closest to the user click
                i = np.argmin(distances)
                rx,ry = centers[i]
                mouse = cnts[i] 

                #the user cannot miss badly
                if pointDistance((rx,ry), (x,y)) > 20:
                    print 'didnt work!'
                    tkMessageBox.showinfo("Alert!", "The position clicked is not close enough to one of my centers!")
                else: 
                    print 'mouse set'
                    ratInitialized = True

        #create window to display the current detection and assign the previous function
        #the deal with the user input
        cv2.namedWindow('initial position', cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('initial position', getMousePosition)

        while ratInitialized == False:
            #read thresholds
            th = cv2.getTrackbarPos('th', window_settings)
            th_max = cv2.getTrackbarPos('max', window_settings)
            #find contours
            cnts = jasf_ratFinder.detectInterestingContours(img, th, th_max, otsu_th) 
            cnts = cnts[0]#we don't need the other outputs 
            #draw
            img2show = myDrawContours(img, cnts)
            #show
            cv2.imshow('initial position', img2show)
            cv2.waitKey(5)

        cv2.destroyWindow('initial position')

        return rx, ry, mouse

def pointDistance(a,b):
    """Compute euclidian distance of two vectors"""
    return np.linalg.norm(np.array(a)-np.array(b))

class MouseNotFound(Exception):
    """Class to be raised in case the mouse is not found"""
    def __init__(self, reason = ''):
        self.reason = reason

def filterMouse(cnts, last_center):
    """ Receives a set of countours and the last known position of the mouse and use a set
    of heuristics to choose the rat """
    rx,ry,mouse = -1,-1,[]
    try:
        if len(cnts) == 0:
            raise MouseNotFound('Detection sees nothing!')
        #--------------------------------
        #find the center that is the closest to the previous detection
        #--------------------------------
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

def main():
    #initialize camera
    cam = cv2.VideoCapture('../../video/avi/myFavoriteVideo.avi')

    #create windows to display output and intermediate results
    window_input = jasf_cv.getNewWindow('input')
    window_otsu = jasf_cv.getNewWindow('otsu')
    window_open = jasf_cv.getNewWindow('open')
    window_small_filter = jasf_cv.getNewWindow('smallFilter')
    window_output = jasf_cv.getNewWindow('output')
    window_settings = jasf_cv.getNewWindow('settings')

    def doNothing(val):
        """function to be passed to createTrackbar"""
        pass

    #create trackbars
    cv2.createTrackbar('th', window_settings, 0, 400, doNothing)
    cv2.createTrackbar('max', window_settings, 100, 400, doNothing)
    cv2.createTrackbar('area_delta', window_settings, 4, 10, doNothing)


    #---------------------------------------------------------------------------
    #initialize required variable
    #---------------------------------------------------------------------------
    rx,ry,mouse = -1, -1,[]
    previous_roi, roi = [], []
    approx, previous_approx = [], []
    #these values were set manually to produce good results
    #alpha is amount of error tolerated when approximating a polynomial surface
    alpha = 34/1000.0
    #this controls the amount of increase in area from one iteration to the other
    allowed_floor_jump = 50
    #---------------------------------------------------------------------------
    #main loop
    #---------------------------------------------------------------------------
    while cam.isOpened():
        #read frame
        ret, frame = cam.read()
        if frame == None:
            print 'finishing due to end of video'
            break
        left, right = devika_cv.break_left_right(frame)

        #read trackbars
        #those two are actually set by the own program
        th = cv2.getTrackbarPos('th', window_settings)
        th_max = cv2.getTrackbarPos('max', window_settings)
        #this one the user can freely set
        area_delta = cv2.getTrackbarPos('area_delta', window_settings)

        #get blue component
        B = right[:,:,0]
        #-----------------------------------------------------------------
        #Step 1: localize the square box
        #-----------------------------------------------------------------
        input = B.copy()
        #we need to keep track of those two previous states
        previous_approx = deepcopy(approx)
        previous_roi = deepcopy(roi)
        #this returns a contourn to the floor region
        #this method also returns the threshold computer by Otsu's method and that should
        #be used later
        roi, approx, inver, otsu_th = jasf_ratFinder.detectFloor(input, 34/1000.0, previous_approx, previous_roi, allowed_floor_jump )
        #make the contour into a mask where ones represent pixels to consider and zeros
        #pixels to disconsider
        floor_mask = np.zeros_like(input)
        floor_mask = cv2.drawContours(floor_mask, [roi], 0, 1, -1)
        #floor_mask = cv2.dilate(floor_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

        #-----------------------------------------------------------------
        #Step 2: find candidates to rat contour
        #-----------------------------------------------------------------
        input = input * floor_mask
        #this will run on the first iteration to initialize the mouse position
        if (rx,ry) == (-1, -1):
            rx,ry,mouse = initialize(input, th, th_max, otsu_th)
            #computer area of the mouse and set the boundaries for the next iteration
            area = cv2.contourArea(mouse)/100
            newTh = max(area - area_delta, 9)
            newTh_max = max(area + area_delta, 16)
            cv2.setTrackbarPos('th', window_settings, int(newTh))
            cv2.setTrackbarPos('max', window_settings, int(newTh_max))

        #find candidates to rat contour
        contours, otsu_threshold, open, filterSmall = jasf_ratFinder.detectInterestingContours(input, th, th_max, otsu_th)

        #-----------------------------------------------------------------
        #Step 3: select which contour is the real mouse
        #-----------------------------------------------------------------
        rx,ry,new_mouse = filterMouse(contours, (rx, ry))
        #if mouse was found, update parameters
        if type(new_mouse) is not bool:
            mouse = new_mouse
            area = cv2.contourArea(mouse)/100
            newTh = max(area - area_delta, 9)
            newTh_max = max(area + area_delta, 16)
            cv2.setTrackbarPos('th', window_settings, int(newTh))
            cv2.setTrackbarPos('max', window_settings, int(newTh_max))

        
        #-----------------------------------------------------------------
        #Step 4: show output and some intermediate results
        #-----------------------------------------------------------------
        #draw countours
        output = input.copy()
        output = myDrawContours(output, [mouse])
        
        cv2.imshow(window_otsu, 255*otsu_threshold)
        cv2.imshow(window_open, 255*open)
        cv2.imshow(window_small_filter, 255*filterSmall)
        cv2.imshow(window_input, input) 
        cv2.imshow(window_output, output) 

        #check if execution should continue or not
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            print 'end of execution due to user command'
            break

    cam.release()
    cv2.destroyAllWindows()

main()
