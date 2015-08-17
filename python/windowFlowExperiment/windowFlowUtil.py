import sys
sys.path.insert(0, '../')
import jasf
from jasf import jasf_cv
sys.path.append('../flowExperiment/')
import flowUtil

import numpy as np
import cv2

import pdb

class WindowFlowComputer(flowUtil.FlowComputer):
    def __init__(self, ws=3):
        self.possible_states = ('waitingInitializing', 'running')
        self.state = self.possible_states[0]
        self.buffer_first, self.buffer_second = list(), list()
        self.windowSize = ws
        self.lk_params = dict(  winSize  = (15,15),\
                                maxLevel = 2,\
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

    def clearBuffers(self):
        self.buffer_first, self.buffer_second = [], []

    def setWindowSize(self, ws):
        self.setWindowSize = ws

    def isReady(self):
        return (len(self.buffer_first) == self.windowSize) and (len(self.buffer_second) == self.windowSize)

    def addToBuffer(self, img):
        if len(self.buffer_first) < 3:
            self.buffer_first.append(img)
        else:
            self.buffer_second.append(img)

    def apply(self, img):
        """If object is not ready to compute because the new window was not received yet, then we return two empty
        np arrays. Check for this by looking at the size of the array. The empty array will have size 0. """
        if self.isReady():
            #take average of buffers then compute flow between then
            #average buffers
            average1 = jasf.cv.averageImages(self.buffer_first)
            average2 = jasf.cv.averageImages(self.buffer_second)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(average1, average2, self.grid, None, **self.lk_params)

            #move new window(second buffer) to old windos(first buffer)
            self.buffer_first = self.buffer_second
            self.buffer_second = []

            # Select good points. That is, points for which the flow was successdully computed
            goodIndex = np.where(st == 1)[0]
            good_new = p1[goodIndex]
            good_old = self.grid[goodIndex]

            cv2.imshow('average1', average1)
            cv2.imshow('average2', average2)

            return good_old, good_new

        else:
            self.addToBuffer(img)
            return np.empty((0,0), dtype = np.uint8), np.empty((0,0), dtype = np.uint8)

if __name__ == "__main__":
    print 'running test mode of WindowFlowComputer...'
    cam = cv2.VideoCapture(0)

    unit = WindowFlowComputer()

    h,w = jasf.cv.getVideoCaptureFrameHeightWidth(cam)
    unit.setGrid(flowUtil.getGrid(w/2,h/2, 100, 100, 10, 10))

    jasf.cv.getManyWindows(['output', 'average1', 'average2'])

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


        cv2.imshow('output', output)


    cv2.destroyAllWindows()
    cam.release()


