import numpy as np
import cv2
from jasf import jasf_cv
from jasf import jasf_ratFinder
import jasf
from devika import devika_cv
from copy import deepcopy
from util import FloorDetectPreProcess
from abstractRatFinder import MouseNotFound


class PreviousCenter_MousePicker(object):
    def __init__(self):
        pass

    def pickCorrectContour(self, cnts, paramsDict):
        """ Receives a set of countours and the last known position of the mouse and use a set
        of heuristics to choose the rat """
        rx,ry,mouse = -1,-1,[]
        last_center = paramsDict['last_center']
        distanceRejectTh = paramsDict['distanceRejectTh']

        try:
            if len(cnts) == 0:
                raise MouseNotFound('Detection sees nothing!')
            #--------------------------------
            #find the center that is the closest to the previous detection
            #--------------------------------
            px,py = last_center
            centers = [jasf_cv.getCenterOfContour(c) for c in cnts]
            distances = [jasf.math.pointDistance(np.array(c) , np.array((px,py))) for c in centers]
            i = np.argmin(distances)
            rx,ry = centers[i]
            mouse = cnts[i] 

            if jasf.math.pointDistance((rx,ry), (px,py)) > distanceRejectTh:
                print 'new center', rx,ry
                print 'previous center', px,py
                raise MouseNotFound('Closest center not close enough to previous position! Is this rat a ninja or what??')

        except MouseNotFound as E:
            print 'rat not found!\n', E.reason
            return last_center[0], last_center[1], False

        return rx,ry,mouse

