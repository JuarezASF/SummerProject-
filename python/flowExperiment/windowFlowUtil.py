""" This file implements the WindowFlowComputer. There are two adjacent windows that slide through the flow of video.
The windows are averaged and we compute the flow between the two averages. The window then move window_length frames to
the right. That is, if at a given moment window 1 corresponds to frames 1,2,3 and window 2 corresponds to frames 4,5,6
then when the window move: window1 contains 4,5,6 and window 2 contains 7,8,9 for a window lenght of 3.
"""
import sys
sys.path.insert(0, '../')
import jasf
from jasf import jasf_cv
import flowUtil

import numpy as np
import cv2

import pdb

class WindowFlowComputer(flowUtil.FlowComputer):
    def __init__(self, ws=3):
        self.setWindowSize(ws)
        self.lk_params = dict(  winSize  = (15,15),\
                                maxLevel = 2,\
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

    def clearAverages(self):
        self.average1 = np.empty((0,0), dtype = np.uint8)
        self.average2 = np.empty((0,0), dtype = np.uint8)
        self.added2average1 = 0
        self.added2average2 = 0

    def setWindowSize(self, ws):
        self.clearAverages()
        self.windowSize = ws

    def isReady(self):
        return (self.added2average1 == self.windowSize) and (self.added2average2 == self.windowSize)

    def addToAverage(self, img):
        if self.added2average1 < self.windowSize:
            if self.added2average1 == 0:
                self.average1 = img * 1.0/self.windowSize
            else:
                self.average1 += img * 1.0/self.windowSize

            self.added2average1 += 1

        else:
            if self.added2average2 == 0:
                self.average2 = img * 1.0/self.windowSize
            else:
                self.average2 += img * 1.0/self.windowSize

            self.added2average2 += 1

    def apply(self, img):
        """If object is not ready to compute because the new window was not received yet, then we return two empty
        np arrays. Check for this by looking at the size of the array. The empty array will have size 0. """
        if self.isReady():
            #take average of buffers then compute flow between then
            #average buffers
            average1 = np.array(self.average1, dtype = np.uint8)
            average2 = np.array(self.average2, dtype = np.uint8)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(average1, average2, self.grid, None, **self.lk_params)

            #move new window(second buffer) to old windos(first buffer)
            self.average1 = self.average2
            self.added2average2 = 0
            self.average2 = np.empty_like((0,0), dtype = np.uint8)

            # Select good points. That is, points for which the flow was successdully computed
            goodIndex = np.where(st == 1)[0]
            good_new = p1[goodIndex]
            good_old = self.grid[goodIndex]

            # debug cv2.imshow('average1', average1)
            # debug cv2.imshow('average2', average2)

            return good_old, good_new

        else:
            self.addToAverage(img)
            return np.empty((0,0), dtype = np.uint8), np.empty((0,0), dtype = np.uint8)

if __name__ == "__main__":
    print 'running test mode of WindowFlowComputer...'
    cam = cv2.VideoCapture(0)

    unit = WindowFlowComputer()

    h,w = jasf.cv.getVideoCaptureFrameHeightWidth(cam)
    unit.setGrid(flowUtil.getGrid(w/2,h/2, 100, 100, 10, 10))

    jasf.cv.getManyWindows(['input', 'output', 'average1', 'average2'])

    ret,frame = cam.read()
    output = np.zeros_like(frame)

    while True:
        ch = cv2.waitKey(30) & 0xFFFF
        if ch == ord('q'):
            break
        ret, frame = cam.read()
        frame = jasf_cv.convertBGR2Gray(frame)

        old, new = unit.apply(frame)
        if old.size != 0:
            output = flowUtil.draw_flow(frame, old, new)


        cv2.imshow('input', frame)
        cv2.imshow('output', output)


    cv2.destroyAllWindows()
    cam.release()
