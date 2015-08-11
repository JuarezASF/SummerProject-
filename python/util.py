import numpy as np
import cv2
import jasf
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
import copy
from copy import deepcopy
from abstractRatFinder import *
import tkMessageBox

class AbstractPreProcessStep:
    def __init__(self):
        pass

    def preProcess(self, img):
        raise Exception('not implemented!')

class FloorDetectPreProcess(AbstractPreProcessStep):
    def __init__(self, alpha = 30/1000.0, alloweJump = 50):
        self.alpha = alpha
        self.allowedJump = 50
        self.roi = []
        self.removeBackground = self.preProcess

    def setParams(self, a=0.003, j=50):
        self.alpha = a
        self.allowedJump = j

    class FailedToFindFloor(Exception):
        """Exception class to be thrown"""
        pass

    def preProcess(self, img, dilatingSize=5, erodeSize=0):
        """
        result, otsu, invert, cnts <- preProcess(img)
        Run a set of filters based on Devika's code, invert it, find contours and
        return the one with the heighest area. Uses countours of previous detection to
        select weather or not the current detection is acceptable. If it is not,return the
        orld countour.
        img: should be a 1 dimensional image
        result: equals to img where pixels are inside the floor region and zero everywhere else
        otsu: threshold used by otsu
        invert: the output of devika's inverted filter(remember to multiply by 255 to visualize it
        cnts: the contours on invert(list of list of points)
        """
        if len(img.shape) > 2:
            raise Exception('this method should receive a gayscale image! convert it first!')

        input = img.copy()
        previous_roi = deepcopy(self.roi)


        roi = []#a list with the points on the floor contour
        otsu_th = 0#the threshould used by otsu
        invert = []#the output of devika's inverted filter

        try:
            #leave Otsu decide the threshold
            otsu_th, otsu_threshold = cv2.threshold(input, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            #opening operation to fill holes and eliminate noise
            open = cv2.morphologyEx(otsu_threshold, cv2.MORPH_OPEN, jasf_ratFinder.detectFloor_cleaning_kernel)
            open = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

            #invert image
            ret, invert = cv2.threshold(open, 0.5, 1, cv2.THRESH_BINARY_INV)

            #use previous contour to reduce what to look at
            if len(previous_roi) > 0:
                invert = cv2.drawContours(invert, [previous_roi], 0, 1, -1)

            #find countours
            ret, cnts, ret2 = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #find approximations to degree determined by epsilon
            approx = [cv2.approxPolyDP(c, self.alpha*cv2.arcLength(c,True), True) for c in cnts] 

            if len(approx) == 0:
                raise self.FailedToFindFloor('Couldn find any interesting area!')

            #the floor is the one with the highest value of area
            roi = approx[np.argmax([cv2.contourArea(c) for c in approx])]

            #analyse if the new detection is acceptable
            #if it is not, return the previous detection as the new one.
            prev_size = len(previous_roi)
            if (prev_size > 0) and ((len(roi) not in range(prev_size-2, prev_size+2+1)) or\
                    (np.abs(cv2.contourArea(roi) - cv2.contourArea(previous_roi)) > self.allowedJump)):
                raise self.FailedToFindFloor('Contours found are all rejected by filter')

            self.roi = roi

        except self.FailedToFindFloor as E:
            self.roi = roi = previous_roi

        #pixels to consider
        floor_mask = np.zeros_like(input)
        floor_mask = cv2.drawContours(floor_mask, [roi], 0, 1, -1)
        if dilatingSize > 0:
            dilateKernel = jasf_cv.getStructuringRectangle(dilatingSize)
            floor_mask = cv2.dilate(floor_mask, dilateKernel)
        if erodeSize > 0:
            erodingKernel = jasf_cv.getStructuringRectangle(erodeSize)
            floor_mask = cv2.erode(floor_mask, cv2.getStructuringElement( cv2.MORPH_RECT, (5,5)))

        return floor_mask*img, otsu_th, invert, cnts

class FloorDetectPreProcess_S_based(AbstractPreProcessStep):
    def __init__(self, alpha = 30/1000.0, alloweJump = 50):
        self.removeBackground = self.preProcess
        self.window_size = 5
        self.th_min = 197
        self.th_max = 227
        self.erodeTimes = 1
        self.dilateTimes = 2

    def setParams(self, winSize=-1, min = -1, max = -1, erode=-1, dilate=-1):
        self.window_size = winSize if winSize != -1 else self.window_size
        self.th_min = min if min != -1 else self.th_min
        self.th_max = max if max != -1 else self.th_max
        self.erodeTimes = erode if erode != -1 else self.erodeTimes
        self.dilateTimes = dilate if dilate != -1 else self.dilateTimes

    class FailedToFindFloor(Exception):
        """Exception class to be thrown"""
        pass

    def preProcess(self, img, equalize = True):
        """ input: This method should receive the S component of the original image.
        Also, remember to cut the image to eliminate the numbers on the bottom. The
        previous two requisites can be obtainedby making img = hsv[:222,:,1], where hsv is
        hsv encoding of the input image.
            output: This method will output the croped version of the input and a mask.
            Usually you want to apply the mask to something else, for example, to the
            original image. Remember that the mask we return is one channel only. You may
            have to convert it to 3 channel. Also, if you want to visualize it, the mask
            contain only zeros and ones, so remember to multiply by 255."""
        if len(img.shape) > 2:
            raise Exception('this method should receive a gayscale image(S channel)! convert it first!')

        s = img.copy()
        if equalize == True:
            s = cv2.equalizeHist(s)

        #read parameters 
        window_size = max(3,self.window_size)
        th_min = max(1, self.th_min)
        th_max = max(1, self.th_max)
        erode_p = self.erodeTimes
        dilate_p = self.dilateTimes

        #compute mean of window around point
        dst = cv2.blur(s, (window_size, window_size))
        #find difference from point to mean
        sub = abs(dst - s)
        #find points in the regions of interest
        mask = cv2.inRange(sub,th_min,th_max)
        #perform two operations of close
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=2)
        #some erode/dilate 
        mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),iterations = erode_p)
        mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),iterations = dilate_p)

        target = mask.copy()

        #find contours in image
        ret, cnts, hier = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #compute the convex hull of the countour of highest area
        hull = cv2.convexHull(cnts[np.argmax([cv2.contourArea(c) for c in cnts])])


        #pixels to consider
        floor_mask = np.zeros_like(img)
        floor_mask = cv2.drawContours(floor_mask, [hull], 0, 1, -1)
        floor_mask = cv2.dilate(floor_mask, cv2.getStructuringElement( cv2.MORPH_RECT, (5,5)))
        return floor_mask*s, floor_mask

def getGBRHSVcomponents(img):
    """Receiva GBR component and return 6 single chanel components"""
    if len(img.shape) != 3:
        raise Exception('this method require a 3 channel BGR input img!')
    G,B,R = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    return G,B,R,H,S,V

class TemplateDetector_abstract(object):
    def __init__(self, name='templateDetector'):
        self.name = name
        self.templates = []
        self.tracking_good = False
        #mark the last known global coordinate of the top left corner of the tracked object
        self.lastKnownPosition = ()
        #index of the last known good template on the self.templates array
        self.lastKnownIndex = -1
        self.failureCount = 0
        self.failureThresh = 50
        self.previous_global_tl = []
        self.previous_global_br = []
        #threshold from above which a match is disconsidered
        self.working_th = 40000
        self.workingTemplates = []

    def track(self, img):
        raise Exception('method not implemented!')

    def failureCountIncrease(self):
        self.failureCount = self.failureCount + 1

    def resetFailureCount(self):
        self.failureCount = 0

    def addTemplate(self, tpl):
        self.templates.append(tpl)
        self.workingTemplates.append(tpl)

    def getLastMatchingTemplate(self):
        return self.templates[self.lastKnownIndex]

    def informSomethingWrongIsHappening(self, problem = 0):
        self.tracking_good = False

    def lookUpOneTemplate(self, img, tpl):
        """receive one monochanel img and one mono channel template. return the min
        differece, and the top_left and bottom_right position of the  rectangle that
        produce that differece. You should use the min_val to decide if the match is any
        good. The top left and bottom right are define in relation to the img position.
        Remember to adjsut this if the img is only a crop of a global larger image."""
        w, h = tpl.shape[::-1]
        res = cv2.matchTemplate(img,tpl,cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return [min_val, top_left, bottom_right]


    def lookUpAllTemplates(self, img, tpls):
        result = []
        for tpl in tpls:
            result.append(self.lookUpOneTemplate(img, tpl))
        return result

    def setWorkingTh(self, val):
        self.working_th = val

    def globalSearch(self, tpls, img, working_th = -1):
        if working_th == -1:
            working_th = self.working_th
        """ tpls shoub be an array of images"""
        matches = self.lookUpAllTemplates(img, tpls)
        errors = [a[0] for a in matches]
        validIndexes = []
        if working_th > 0:
            validIndexes = [i for i in range(len(tpls)) if errors[i] < working_th]
        else:
            validIndexes = [i for i in range(len(tpls))]

        if len(validIndexes) == 0:
            raise NoTemplatesFound()

        accpted_errors = [errors[i] for i in validIndexes]
        tl = [tuple(matches[i][1]) for i in validIndexes]#top left
        br = [tuple(matches[i][2]) for i in validIndexes]#bottom right

        bestIndex = np.argmin(errors)
        best_tl   = matches[bestIndex][1]


        return tl, br, accpted_errors, validIndexes, bestIndex, best_tl

    def simpleGlobalSearch(self, tpls, img):
        pass


class TemplateDetector(TemplateDetector_abstract):
    def __init__(self, name='corner'):
        super(TemplateDetector, self).__init__(name)

    def setWindowSize(self, ws):
        self.window_size = ws

    def convertLocalToGlobal(self, point_local, topLeft_global):
        out = (np.array(point_local) + np.array(topLeft_global)).tolist()
        return tuple(out)


    def getSmallWindow(self, img, square_size = 32):
        """returns the roi and its top left position in relationship to the global image."""
        l = square_size/2
        w, h = self.templates[self.lastKnownIndex].shape[::-1]
        #set search area around last known position
        x,y = self.lastKnownPosition
        x = x + w/2
        y = y + h/2
        top_left_x = max(0, x-l)
        top_left_y = max(0, y-l)
        #this will be required to convert local findings to global position
        roi_top_left = (top_left_x, top_left_y)

        bottom_right_x = min(img.shape[1], x+l)
        bottom_right_y = min(img.shape[0], y+l)

        roi = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        return roi, roi_top_left

    def track(self, img):
        """should receive a mono chanel image"""
        if len(self.templates) == 0:
            return False, False, False
        match = []
        global_tl = ()
        global_br = ()

        try:
            if self.tracking_good == True:
                roi, roi_tl = self.getSmallWindow(img, square_size = 48)

                #look for the current working templates around a box cetered a the previous best position
                local_tl, local_br, acc_errors, valid_indexes, bestIndex, local_best_tl = self.globalSearch(self.workingTemplates, roi)

                if len(acc_errors) == 0 or min(acc_errors) > self.working_th:
                    #look for all templates on roi
                    local_tl, local_br, acc_errors, valid_indexes, bestIndex, local_best_tl = self.globalSearch(self.templates, roi)
                    if len(acc_errors) == 0 or min(acc_errors) > self.working_th:
                        raise NoTemplatesFound()
                    else:
                        #detection of all templates inside roi was a success
                        #only good templates are kept
                        self.workingTemplates = [self.templates[i] for i in valid_indexes]

                else:
                    #detection of previous templates was successful
                    #only good templates are kept
                    self.workingTemplates = [self.workingTemplates[i] for i in valid_indexes]

                
                #convert local coordinates to global one
                global_tl = [self.convertLocalToGlobal(a, roi_tl) for a in local_tl]
                global_br = [self.convertLocalToGlobal(a, roi_tl) for a in local_br]
                #update best position
                global_best_tl = self.convertLocalToGlobal(local_best_tl, roi_tl)
                self.lastKnownPosition = global_best_tl
            
            else:
                #look for all templates in the entire image
                print 'realizing global search'
                global_tl, global_br, acc_errors, valid_indexes, bestIndex, global_best_tl = self.globalSearch(self.templates, img)
                self.tracking_good = True
                self.workingTemplates = [self.templates[i] for i in valid_indexes]
                self.lastKnownPosition = global_best_tl

        except NoTemplatesFound as E:
            print 'next round will reset search for corner', self.name
            self.tracking_good = False
            self.failureCountIncrease()
            if self.failureCount > self.failureThresh:
                raise CornerWasLostException(self.name)
            return self.previous_global_tl, self.previous_global_br, True
        
        self.previous_global_tl = global_tl 
        self.previous_global_br = global_br 

        return global_tl, global_br, True


class NoTemplatesFound(Exception):
    def __init__(self, msg=""):
        self.msg = msg
class CornerWasLostException(Exception):
    def __init__(self, cornerName):
        self.name = cornerName 

class MovingTemplateDetector(TemplateDetector_abstract):
    def __init__(self, name):
        super(MovingTemplateDetector, self).__init__(name)

    def track(self, img):
        """should receive a mono chanel image"""
        if len(self.templates) == 0:
            return False, False, False
        try:
            #look for all templates in the entire image
            global_tl, global_br, acc_errors, valid_indexes, bestIndex, global_best_tl = self.globalSearch(self.templates, img, -100)

            #update templates 
            if len(global_tl) > 0:
                self.templates = []
                for i,_ in enumerate(global_tl):
                    tl = global_tl[i]
                    br = global_br[i]
                    tpl = img[tl[1]:br[1], tl[0]:br[0]]
                    self.templates.append(tpl)

        except NoTemplatesFound as E:
            return self.previous_global_tl, self.previous_global_br, True

        self.previous_global_tl = global_tl 
        self.previous_global_br = global_br 

        return global_tl, global_br, True



class DetectMovement:
    globalMovingStatus_normal = 0
    globalMovingStatus_backMovement = 1
    globalMovingStatus_camera_move = 2

    def __init__(self, img):
        self.prev = img.copy()
        self.shape = img.shape[:]
        self.grid = []
        self.x_step = 40
        self.y_step = 40
        self.computeGrid()
        self.lk_params = dict(  winSize  = (15,15),\
                                maxLevel = 2,\
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
        self.movingTh = 2
        self.warningTh = 0.15
        self.dangerTh = 0.85
        
        self.controlWaitingStabilize = False
        self.controlWaitingSteps = 0
        self.lastState = 0

        self.waitTime = 200

    def setSteps(self, sx, sy):
        self.x_step = sx
        self.y_step = sy
        self.computeGrid()

    def setWaitTime(self, time):
        self.waitTime = time

    def setMovingTh(self, th):
        self.movingTh = th

    def setDangerTh(self, th):
        self.dangerTh = th

    def setWarningTh(self, th):
        self.warningTh = th

    def computeGrid(self):
        grid = []
        for i in range(5,self.shape[1]-5, self.x_step):
            for j in range(5,self.shape[0]-5, self.y_step):
                grid.append([[i,j]])

        self.grid = np.array(grid, dtype=np.float32)


    def detect(self, img):
        """ret = 0 if nothing is moving, 1 if someone is moving in the back and 2 if the
        camera is moving """
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, img, self.grid, None, **self.lk_params)
        self.prev = img.copy()

        # Select good points
        good_new = p1[st==1]
        good_old = self.grid[st==1]

        #compute displacement
        displacements = [jasf.math.pointDistance(good_new[i], good_old[i]) for i in range(len(good_new))]

        movingPoints = [(good_old[i], good_new[i]) for i in range(len(good_old)) if displacements[i] > self.movingTh]

        ratio = len(movingPoints)/float(len(good_old))

        ret = 0
        if ratio > self.warningTh:
            ret = 1
            if ratio > self.dangerTh:
                ret = 2

        if ret > 0:
            self.controlWaitingSteps = 60
            self.controlWaitingStabilize = True
            self.lastState = ret

        if self.controlWaitingStabilize == True:
            self.controlWaitingSteps = self.controlWaitingSteps - 1
            ret = self.lastState
            if self.controlWaitingSteps <= 0:
                self.controlWaitingStabilize = False
                ret = 0

        return movingPoints, ret

class ImageSplitter(object):
    def __init__(self):
        pass

    def getLeftRight(self, img, lr):
        """receive img of any type and lr should be a string saying 'left' or 'right' """
        pass

