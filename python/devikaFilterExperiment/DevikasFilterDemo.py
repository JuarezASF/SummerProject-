import sys
sys.path.insert(0, '../')
sys.path.append('./flowExperiment')
import numpy as np
import cv2
import jasf
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
import copy
from copy import deepcopy
from AbstractDemo import AbstractDemo
import tkMessageBox
from util import *
from DevikasFilterGroundSubtraction_ContourFinder import DevikasFilterGroundSubtraction_ContourFinder
from AbstractDemo import MouseNotFound
from PreviousCenter_MousePicker import PreviousCenter_MousePicker
from flowComputer import FlowComputer


class VideoInvalid(Exception):
    def __init__(self, reason = ''):
        self.reason = reason

class mouseDescription:
    def __init__(self):
        self.rx = -1
        self.ry = -1
        self.mouse = []
        self.ratInitialized = False

class DevikasFilterDemo(AbstractDemo):

    def __init__(self, GUI):
        super(DevikasFilterDemo, self).__init__()
        """Should receive a GUI reference so the class can output the video"""
        self.GUI = GUI
        self.cam = None
        self.contourFinder = DevikasFilterGroundSubtraction_ContourFinder()
        self.contourPicker = PreviousCenter_MousePicker()

        def doNothing(val):
            pass

        self.setMousePosition(-1,-1,[])
        #-----------------------------------------------------------------
        #Initialize trackbars
        #-----------------------------------------------------------------
        self.GUI.addSettingsPropertie('th', 0, 400, 0)
        self.GUI.addSettingsPropertie('max', 0, 400, 100)
        self.GUI.addSettingsPropertie('delta', 0, 10, 4)
        self.GUI.addSettingsPropertie('dilateSize', 0, 21, 5)
        self.GUI.addSettingsPropertie('erodeSize', 0, 21, 0)
        self.GUI.addSettingsPropertie('LRA', 0, 2, 1)

    def setMousePosition(self, rx, ry, mouse):
        self.rx = rx
        self.ry = ry 
        if len(mouse) > 0:
            self.mouse = copy.deepcopy(mouse)
    
    def initialize(self, videoFile):
        """ This function will set self.cam to the videoFile taken and intialize the mouse
        position. If the algorithm is in doubt about the mouse position, it will prompt the
        user to inform it with a mouse click"""
        print 'initializing'
        #initialize camera
        if self.cam != None:
            if self.cam.isOpened():
                self.cam.release()
        self.cam = cam = cv2.VideoCapture(videoFile)

        if not cam.isOpened():
            raise Exception('Video could not be opened! Is path correct?')
        self.flowComputer = FlowComputer()

        #-----------------------------------------------------------------
        #read first frame and find initial position of mouse
        #-----------------------------------------------------------------
        self.askForUserInput()

    def askForUserInput(self):
        #read image and break into right and left
        ret, img = self.cam.read()
        left, right = devika_cv.break_left_right(img)
        inputList = [left, right, img]

        #read parameters from settings window
        settings = self.readSettingsParameters()
        th, th_max, area_delta, dilateSize, erodeSize, LRA = settings['th'], settings['max'], settings['delta'],\
                settings['dilateSize'], settings['erodeSize'], settings['LRA']

        #select which image to use
        input = inputList[LRA]

        #get blue component
        input = input[:,:,1]

        #ask user to select contour
        self.askUserForInput_selectContour(input)

        #compute area and set the new threshoulds
        self.updatedValuesOfTh(area_delta)


    def askUserForInput_selectContour(self,img):
        def readParametersAndExecute(event):
            #read parameters from settings window
            settings = self.readSettingsParameters()
            th, th_max, area_delta, dilateSize, erodeSize = settings['th'], settings['max'], settings['delta'],\
                    settings['dilateSize'], settings['erodeSize']
            #find contours
            self.contourFinder.setParams(dilateSize, erodeSize, th, th_max)
            cnts, otsu_threshold, filterSmall = self.contourFinder.detectInterestingContours(img)
            #draw all contours
            img2show = jasf_cv.drawContours(img, cnts)

            #show image with contours produced with the new parameters to the user 
            self.GUI.setUserClickWindowImg(img2show)
            #ret val will be true if the user clicked on the screen
            pointRead, retVal = self.GUI.readUserInputClick()
            if retVal == False:
                return
            x,y = pointRead[0], pointRead[1]
            self.GUI.setClickWasRead()

            if len(cnts) == 0:
                self.GUI.log("the current parameters produce no contour! you need to tune it!")
                return

            #compute center of current contours
            centers = [jasf_cv.getCenterOfContour(c) for c in cnts]
            #compute distances from centers to user click
            distances = [jasf.math.pointDistance(np.array(c), np.array((x,y))) for c in centers]
            #find closes center
            i = np.argmin(distances)
            rx,ry = centers[i]
            mouse_cnt = cnts[i] 
            #reject distant points
            if jasf.math.pointDistance((rx,ry), (x,y)) > 20:
                self.GUI.log('didnt work!')
                tkMessageBox.showinfo("Alert!", "The position clicked is not close enough to one of my centers!")
            else: 
                #this is the case of sucessfull detection
                self.setMousePosition(rx, ry, mouse_cnt)
                self.GUI.log('mouse set')
                #by closing the window we proceed with the code
                self.GUI.setUserClick_Done()

        clickWindow = self.GUI.openUserClickWindow(readParametersAndExecute)
        readParametersAndExecute(0)
        self.GUI.master.wait_window(clickWindow)

    def updatedValuesOfTh(self, area_delta):
        """update the values of threshold based on the area of the current self.mouse"""
        #compute area and set the new threshoulds
        area = cv2.contourArea(self.mouse)/100
        newTh = max(area - area_delta, 0)
        newTh_max = max(area + area_delta, 16)
        self.GUI.setSettingsPropeitie('th', newTh)
        self.GUI.setSettingsPropeitie('max', newTh_max)

    def finish(self):
        self.cam.release()

    def readSettingsParameters(self):
        """Here is how to use it:
        settings = self.readSettingsParameters()
        th, th_max, area_delta, dilateSize, erodeSize = settings['th'], settings['max'], settings['delta'],\
                settings['dilateSize'], settings['erodeSize']
        """
        toRead = ['th', 'max', 'delta', 'dilateSize', 'erodeSize', 'LRA']
        settings = self.GUI.readManySettingsPropertie(toRead)

        return settings

    def update(self):
        #read frame
        ret, frame = self.cam.read()
        if ret == False:
            raise VideoInvalid('finishing due to end of video')
        left, right = devika_cv.break_left_right(frame)
        inputList = [left, right, frame]

        #read parameters from settings window
        settings = self.readSettingsParameters()
        th, th_max, area_delta, dilateSize, erodeSize, LRA = settings['th'], settings['max'], settings['delta'],\
                settings['dilateSize'], settings['erodeSize'], settings['LRA']


        #select which image to use
        input = inputList[LRA]
        h,w,d = input.shape
        ho, wo = 40,40
        extendedInput = np.zeros((h+2*ho,w+2*wo,d), dtype=np.uint8)
        extendedInput[ho:ho+h, wo:wo+w, :] = input[:,:,:]

        #get blue component(as suggested by Devika)
        B = input[:,:,0]
        Bextended = extendedInput[:,:,0]

        #-----------------------------------------------------------------
        #find candidates to rat contour
        #-----------------------------------------------------------------
        input = B.copy()
        self.contourFinder.setParams(dilateSize, erodeSize, th, th_max)
        contours, otsu_threshold, filterSmall = self.contourFinder.detectInterestingContours(input)

        #-----------------------------------------------------------------
        #Step 3: select which contour is the real mouse
        #-----------------------------------------------------------------
        rx,ry,new_mouse = self.contourPicker.pickCorrectContour(contours, {'last_center':(self.rx, self.ry), 'distanceRejectTh':2000})
        #if mouse was found, update parameters
        if type(new_mouse) is not bool:
            self.setMousePosition(rx,ry,new_mouse)
            self.updatedValuesOfTh(area_delta)

        #-----------------------------------------------------------------
        #Step 4: show output and some intermediate results
        #-----------------------------------------------------------------
        #draw countours
        output = Bextended.copy()
        #convert mouse coordinates to extended frame
        offset = np.empty_like(self.mouse)
        offset.fill(40)
        translatedMouse = self.mouse + offset
        #draw fixed dim rectangle around mouse
        output = jasf_cv.drawFixedDimAroundContourCenter(output, [translatedMouse], (200, 0, 200), np.array((60,60)))
        #get fixed lenght rectangle image
        mouseImg = jasf.cv.getRoiAroundContour(extendedInput, translatedMouse, dim = np.array((60,60)))

        mousePB = jasf_cv.convertBGR2Gray(mouseImg)
        oldP, newP = self.flowComputer.apply(mousePB)

        self.GUI.setImg(B, 0)
        self.GUI.setImg(255*otsu_threshold, 1)
        self.GUI.setImg(255*filterSmall, 2)
        self.GUI.setImg(output, 3)
        self.GUI.setImg(mouseImg, 4)
