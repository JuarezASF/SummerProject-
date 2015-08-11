import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from copy import deepcopy
from util import FloorDetectPreProcess
from AbstractInterestingContourFinder import AbstractInterestingContourFinder

class DevikasFilterGroundSubtraction_ContourFinder(AbstractInterestingContourFinder):
    def __init__(self):
        self.floorFilter = FloorDetectPreProcess()
        self.dilateSize = 0
        self.erodeSize = 0
        self.th, self.th_max = -1, -1

    def setParams(self, dilate, erode, th, th_max):
        self.dilateSize = dilate
        self.erodeSize = erode
        self.th = th
        self.th_max = th_max

    def methodHeuristic(self, input, otsu_th):
        """Apply a set of filtering techniques suggested by Devika and return contours and
        some of the images produced in the process so one can visualize outputs of the
        filters. Small objects are removed. Only objects with area between th and th_max
        are candidates to be returned. Chech function areaBandPassObjectFilter to learn
        more."""
        #use threshould informed bu user. This threshould was probably computer before by otsu
        #otsu_threshold is a binnary image
        ret, otsu_threshold = cv2.threshold(input, otsu_th, 1, cv2.THRESH_BINARY)
        #opening operation to fill holes and eliminate noise
        open = cv2.morphologyEx(otsu_threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        open = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        #filter small objects
        filterSmall = jasf_cv.areaBandPassObjectFilter(open.copy(), self.th, self.th_max)
        #find contours
        target = filterSmall.copy()
        output, contours, hier = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, otsu_threshold, open, filterSmall

    def detectInterestingContours(self, input):
        """Input image should be a grayscale one"""
        backgroundmadeblack, otsu_th, invert, cnts = self.floorFilter.removeBackground(input, self.dilateSize, self.erodeSize)

        #find candidates to rat contour
        input = backgroundmadeblack
        contours, otsu_threshold, open, filterSmall = self.methodHeuristic(input, otsu_th)

        return contours, otsu_threshold, filterSmall

