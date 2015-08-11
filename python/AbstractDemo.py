#!/bin/
import numpy as np
import cv2
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from copy import deepcopy

class MouseNotFound(Exception):
    """Class to be raised in case the mouse is not found"""
    def __init__(self, reason = ''):
        self.reason = reason

class AbstractDemo(object):
  def __init__(self, GUI = None):
      self.preProcessSteps = []

  def initialize(self, videoFile):
    raise Exception('Method not Implemented!')

  def finish(self):
    raise Exception('Method not Implemented!')

  def update(sef, img):
    raise Exception('Method not Implemented!')

  def askForUserInput(self):
    raise Exception('Method not Implemented!')
