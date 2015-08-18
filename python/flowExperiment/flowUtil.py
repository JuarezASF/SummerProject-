import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
import matplotlib.pyplot as plt

import jasf
from jasf import jasf_cv
from devika import devika_cv

class FlowComputer(object):
    """ This class allow you compute the optical flow on a regular rectangular grid around the image. The only
    parameters are the spacing in x and y of the grid points. The grid starts at pizel (5,5) and ends at (L-5,M-5)"""
    def __init__(self):
        """ constructor will initialize a default mesh. Please set desired steps with setSteps before use"""
        self.lk_params = dict(  winSize  = (15,15),\
                                maxLevel = 2,\
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

        self.prev = np.empty((0,0), dtype = np.int8)

    def setGrid(self, g):
        self.grid = g

    def apply(self, img, paramsDict=dict()):
        """ret = 0 if nothing is moving, 1 if someone is moving in the back and 2 if the
        camera is moving """
        if self.prev.shape == (0,0):
            self.prev = img.copy()
            return self.grid, self.grid
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, img, self.grid, None, **self.lk_params)
        self.prev = img.copy()

        # Select good points. That is, points for which the flow was successdully computed
        goodIndex = np.where(st == 1)[0]
        good_new = p1[goodIndex]
        good_old = self.grid[goodIndex]

        return good_old, good_new

def getGrid(x,y,w,h,x_step=1, y_step=1):
    """return a grid in the proper format to be used by optical flow computation"""
    X,Y = np.mgrid[x:x+w:x_step, y:y+h:y_step]
    return np.array(np.vstack((X.flatten(),Y.flatten())).transpose(), dtype=np.float32) 

def draw_flow(img, pts, next_pts, flowColor = (0,0,255), flowThickness = 1, p=1, q=1, th = 0, drawArrows=False,
        lenghtOfArrayArm = 2, angleOfArrow=np.pi/3):
    """ Draw p every q flow points. Flow is draw only if its magnitude is higher than th"""
    if pts.shape[0] == 0 or next_pts.shape[0] == 0 or pts.shape[0] != next_pts.shape[0]:
        return img
    lines = np.hstack((pts, next_pts))
    #make it into format opencv wants
    lines = lines.reshape(-1,2,2)
    #round up to nears integer
    lines = np.int32(lines + 0.5)

    #select p every q
    index = np.arange(lines.shape[0])
    index = index[(index%q) < p]
    lines = lines[index]

    #filter small values
    if th > 0:
        #make points into a easy way to manipulate
        points = lines.reshape(-1, 4)
        #compute displacement
        displacement = points[:,2:4] - points[:,0:2]
        S = np.linalg.norm(displacement, axis=1)
        lines = lines[S > th]
    
    if len(img.shape) < 3:
        #make sure we're dealing with a BGR image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #draw multiple lines
    cv2.polylines(img, lines, isClosed = False, color = flowColor, thickness=flowThickness)

    import pdb
    if drawArrows:
        #compute flow direction
        flow = lines[:, 1, :] - lines[:,0,:]
        flow_angle = np.arctan2(flow[:,1], flow[:,0]).reshape(-1,1)

        #get start point of every arrow
        startPoints_x = lines[:, 1, 0].reshape(-1,1)
        startPoints_y = lines[:, 1, 1].reshape(-1,1)

        #get end point of arrow arm 1
        endPoints_x = (startPoints_x + lenghtOfArrayArm * np.cos( angleOfArrow + np.pi + flow_angle)).reshape(-1,1)
        endPoints_y = (startPoints_y + lenghtOfArrayArm * np.sin( angleOfArrow + np.pi + flow_angle)).reshape(-1,1)

        #get end point of arrow arm 2
        endPoints2_x = (startPoints_x + lenghtOfArrayArm * np.cos( -1.0*angleOfArrow + np.pi + flow_angle)).reshape(-1,1)
        endPoints2_y = (startPoints_y + lenghtOfArrayArm * np.sin( -1.0*angleOfArrow + np.pi + flow_angle)).reshape(-1,1)


        #create array with line indications the way opencv wants it
        arrowArms  = np.hstack((startPoints_x, startPoints_y, endPoints_x,  endPoints_y))
        arrowArms2 = np.hstack((startPoints_x, startPoints_y, endPoints2_x, endPoints2_y))
        arrowArms = np.vstack((arrowArms, arrowArms2))
        arrowArms =  arrowArms.reshape((-1,2,2))
        arrowArms = np.array(arrowArms, dtype = np.int32)


        #draw multiple lines
        cv2.polylines(img, arrowArms, isClosed = False, color = flowColor, thickness=flowThickness)


    return img

class FlowFilter(object):
    """ Basic flow filter that selects flow vectors with magnitude within a range """
    def __init__(self):
        self.low_th = -1
        self.upper_th = -1

    def setTh(self, low, upper):
        self.low_th = low
        self.upper_th = upper

    def apply(self, oldP, newP):
        flowNorm = np.linalg.norm(newP - oldP, axis = 1)
        valid = np.where(np.logical_and(flowNorm >= self.low_th, flowNorm <= self.upper_th))
        return oldP[valid], newP[valid]

class FlowFilter_ConnectedRegions(FlowFilter):
    """ This class implements a filter of optical flow by connectivity. If a flow appears aline it is likely that it is
    spurious data. The filtering is done by paiting an image with white in the points where there is flow, then blurring
    it with a dilate operation(in order to connect close but not adjacent points), findig countourns and then band pass
    filtering those countourns by area. Only flow vectors inside a valid contour are kept.
    """
    def __init__(self, shape=None):
        """ 
        The shape should be of the form (h,w). The shape is necessary to create a mask of appropriate size when
        filtering flow vectors
        """
        super(FlowFilter_ConnectedRegions, self).__init__()
        self.img_shape = shape

    def apply(self, oldP, newP, debugMode=False):
        oldP = oldP.astype(np.int)
        #we need to test for this so to avoid bugs
        if oldP.shape[0] == 0:
            return oldP[[]], newP[[]]
        #find demension to create image
        h = oldP[:,1].max() + 1
        w = oldP[:,0].max() + 1
        if debugMode:
            h,w = self.img_shape
        #we need to test for this so to avoid bugs
        if (h == 0) or (w == 0):
            return oldP[[]], newP[[]]

        #initialize matrixes
        grid = np.zeros((h,w), dtype = np.uint8)
        out  = grid.copy()

        #paint every point that has a flow vector with white
        #remember point coordinates and their position in the grid are inverted
        grid[oldP[:,1], oldP[:,0]] = 255

        #smooth image so we don't need pixels to be actually adjacent in order to close contour
        s = 7
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s,s)), iterations = 1)

        #find contour of blured image
        #the returned cnts is a list of np.arrays of shape (-1,1,2)
        #we pass a copy of grid because the function will modify the input
        out__, cnts, hier = cv2.findContours(grid.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #we need to test for this so to avoid bugs
        if len(cnts) == 0:
            return oldP[[]], newP[[]]

        #compute area of contours
        #this is not done in parallel, but this is a cheap operation and hopefully not that expensive
        cntsArea = np.array([cv2.contourArea(x) for x in cnts], dtype=np.int)

        #find valid indexes
        valid = np.where(np.logical_and(cntsArea >= self.low_th, cntsArea <= self.upper_th))
        #again we test to avoid bugs
        if len(valid) == 0:
            return oldP[[]], newP[[]]

        #grab valid contours
        cnts = np.array(cnts)
        cnts = cnts[valid]

        #paint filled contours with white and with a grey border
        cv2.drawContours(out, cnts, -1, 150, 2)
        cv2.drawContours(out, cnts, -1, 255, -1)

        #find indexes of flows where the corresponding mask is white
        valid = np.where(out[oldP[:,1], oldP[:,0]] == 255)

        if debugMode:
            cv2.imshow('flow filter debug grid', grid)
            cv2.imshow('flow filter debug mask', out)

        return oldP[valid], newP[valid]


if __name__ == "__main__":
    cam = cv2.VideoCapture('../../video/mp4/2014-07-16_13-25-12.mp4')

    ret, frame = cam.read()
    flowComputer = FlowComputer(frame.shape)
    flowComputer.setSteps(5,5)

    cv2.namedWindow('input', cv2.WINDOW_NORMAL)
    while True:
        ch = jasf.cv.waitKey(5)
        if ch == ord('q'):
            break
        ret, frame = cam.read()

        old, new = flowComputer.apply(frame)

        show = draw_flow(frame, old, new, th=0.5)

        if ret == False:
            break

        cv2.imshow('input', frame)


    cv2.destroyAllWindows()
    cam.close()
