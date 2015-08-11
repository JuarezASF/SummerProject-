import numpy as np
import cv2
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from copy import deepcopy

class AbstractInterestingContourFinder(object):
    def __init__(self):
        raise Exception('Method not Implemented!')

    def detectInterestingContours(self, img):
        raise Exception('Method not Implemented!')
