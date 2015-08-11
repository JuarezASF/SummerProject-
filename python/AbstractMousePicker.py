import numpy as np
import cv2
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from copy import deepcopy
from util import FloorDetectPreProcess


class AbstractMousePicker(object):
    def __init__(self):
        raise Exception('Method not Implemented!')

    def pickCorrectContour(self, cnts, paramsDict):
        """<- rx, ry, cnt """
        raise Exception('Method not Implemented!')

