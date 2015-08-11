#! /usr/bin/python
import sys
sys.path.insert(0, '../')

import numpy as np
import cv2
from jasf import jasf_cv
from devika import devika_cv


    
def draw_sparse_flow(img, pts, next_pts):
    x = [p[0][0] for p in pts]
    y = [p[0][1] for p in pts]

    x2 = [p[0][0] for p in next_pts]
    y2 = [p[0][1] for p in next_pts]

    #get flow at those points
    lines = np.vstack([x, y, x2, y2]).T.reshape(-1, 2, 2)
    #round up to nears integer
    lines = np.int32(lines + 0.5)
    #make sure we're dealing with a BGR image
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #draw multiple lines
    cv2.polylines(vis, lines, isClosed = False, color = (0, 0, 255), thickness=1)
    #draw circles 
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

def draw_hsv(flow):
    
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,1] = 255
    
    fx, fy = flow[...,0], flow[...,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    
    # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,2] = np.minimum(v*4,255)
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    return rgb

def pickRightCenter(centers):
    right = []
    for c in centers:
        v = c[2]
        if v > 900:
            right.append(c)
    if len(right) == 0:
        print 'Nothing is Moving!'
        return None
    elif len(right) == 1:
        return right[0]
    else:
        print 'More than one thing is moving!'
        return None
   

window_flowMagnitude = "FlowMagnitude"
window_output = "Output"
window_flow = "Flow" 

cv2.namedWindow(window_flowMagnitude, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_output, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_flow, cv2.WINDOW_NORMAL)

cv2.moveWindow(window_flowMagnitude, 0, 0)
cv2.moveWindow(window_output, 260, 0)
cv2.moveWindow(window_flow, 520, 0)


#initialize camera
from config import video2load
cam = cv2.VideoCapture(video2load)
#obtain first frame
ret, prev = cam.read()
left,prev = devika_cv.break_left_right(prev)
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

#define filter parameters
noiseFilterKernelSize = (7,7)

#for paiting the hsv field
hsv = np.zeros_like(prev)
hsv[...,1] = 255

#criteria for kmeans
my_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)

#initialize kalman filter
kalman = cv2.KalmanFilter(dynamParams = 2, measureParams = 2, controlParams = 0)
kalman.statePost = np.array([10, 10])
kalman.measurementMatrix = np.array([[1,0],[0,1]],np.float32)
kalman.transitionMatrix = np.array([[1,0],[0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.9

while (cam.isOpened()):
    for i in range(5):
        ret, img = cam.read()
    #reset control variables
    control_detectionSucess = False
    control_moving_camera   = False

    #pre-process image obtained to reduce noise
    left,img = devika_cv.break_left_right(img)#for now we care only about half the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))

    #compute optical flow
    flow = jasf_cv.computeDenseOpticalFlow(prevgray, gray)
    prevgray = gray

    #obtain mag and ang of flow and process it
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    _,mag = cv2.threshold(src=mag, thresh=1.0, maxval=0.1, type=cv2.THRESH_TOZERO)

    mag = cv2.erode(mag,(7,7),iterations = 10)
    mag = cv2.dilate(mag,(5,5),iterations = 1)

    if control_moving_camera == False:
        #extract features from optical flow
        f0, f1, f2 = [], [], []
        for i,row in enumerate(gray):
            for j,elem in enumerate(row):
                f0.append(j)
                f1.append(i)
                f2.append(1000*mag[i][j])
        Z = np.vstack((f0, f1, f2))
        Z = np.float32(Z.transpose())

        #call kmeans
        if len(Z) < 5:
            control_detectionSucess = False
        else:
            ret,label,center=cv2.kmeans(\
                data=Z, K=2, bestLabels=None, criteria = my_criteria,\
                attempts=5, flags=cv2.KMEANS_PP_CENTERS)
            mice_measured = pickRightCenter(center)
            if mice_measured != None:
                control_detectionSucess = True
            
        #kalman predicts
        if control_detectionSucess == True:
            kalman.correct(mice_measured[0:2]) 
        mice = kalman.predict()

        #draw rectangles around center
        mice = [int(x) for x in mice]
        cx,cy = mice[0], mice[1]
        w,h = 40,32
        track_window = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        x,y,x2,y2 = track_window
        cv2.rectangle(img, (x, y), (x2, y2), 255, 2)


    #show images
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow(window_flowMagnitude,rgb)

    cv2.imshow(window_output,img)
    cv2.imshow(window_flow, devika_cv.draw_flow(gray, flow))

    #decide weather to continue or not
    ch =  cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
