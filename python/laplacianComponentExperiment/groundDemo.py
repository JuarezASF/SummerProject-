import sys
sys.path.insert(0, '../')
import Tkinter as tk
import pygubu
from AbstractDemo import *
import cv2
import numpy as np
import Image, ImageTk
from util import *

from TkOpenCV import *


class groundFinder(abstractRatFinder):
    def __init__(self, GUI = None):
        """Should receive a GUI reference so the class can output the video"""
        self.GUI = GUI
        self.cam = None
        #-----------------------------------------------------------------
        #Initialize trackbars
        #-----------------------------------------------------------------
        self.GUI.addSettingsPropertie('window_size', 0, 13, 5)
        self.GUI.addSettingsPropertie('th_min', 0, 255, 12)
        self.GUI.addSettingsPropertie('th_max', 0, 255, 220)
        self.GUI.addSettingsPropertie('erode', 0, 13, 5)
        self.GUI.addSettingsPropertie('dilate', 0, 13, 3)
        self.GUI.addSettingsPropertie('LR', 0, 2, 0)

    def initialize(self, videoFile):
        #initialize camera
        if self.cam != None:
            if self.cam.isOpened():
                self.cam.release()
        self.cam = cam = cv2.VideoCapture(videoFile)
        self.filter = FloorDetectPreProcess_S_based()

        if not cam.isOpened():
            raise Exception('Video could not be opened! Is path correct?')

    def finish(self):
        self.cam.release()

    def readSettingsParameters(self):
        win     = self.GUI.readSettingsPropertie('window_size')
        min     = self.GUI.readSettingsPropertie('th_min') 
        max     = self.GUI.readSettingsPropertie('th_max')
        erode   = self.GUI.readSettingsPropertie('erode')
        dilate  = self.GUI.readSettingsPropertie('dilate')
        LR      = self.GUI.readSettingsPropertie('LR')

        return win, min, max, erode, dilate, LR

    def update(self):
        #read frame
        ret, frame = self.cam.read()
        if ret == False:
            raise VideoInvalid('finishing due to end of video')
        left, right = devika_cv.break_left_right(frame)

        #read parameters from settings window
        win, min, max, erode, dilate, LR = self.readSettingsParameters()
        inputList = [left, right,frame]
        input = inputList[LR]
        hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        self.filter.setParams(win, min, max, erode, dilate)

        s, mask = self.filter.preProcess(hsv[:222,:,1], equalize=False)
        output = gray[:222,:]*mask

        ret, output = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY)
        open = cv2.morphologyEx(output, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        open = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        
        self.GUI.setImg(input, 0)
        self.GUI.setImg(output, 1)
        self.GUI.setImg(s, 2)
        self.GUI.setImg(open, 3)

from GUI import RatFinderGUI

if __name__ == '__main__':
    root = tk.Tk()
    app = RatFinderGUI(root)
    ratFinder = groundFinder(app)
    app.setRatFinder(ratFinder)

    app.addVideo("../video/mp4/selected/rightMouseWalks2TheBack.mp4","moving rat")
    app.addVideo("../video/mp4/selected/walkingInTheBack.mp4",'someone walking in the back')
    app.addVideo('../video/mp4/selected/camereMoves.mp4','camera moving')
    app.addVideo("../video/mp4/selected/camereMoves2.mp4", 'camera move 2')

    
    root.after(1, app.executionLoop)
    root.mainloop()
