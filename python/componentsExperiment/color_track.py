#20 mean shift algorithm
import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from jasf import jasf_cv

    
from config import video2load
cap = cv2.VideoCapture(video2load)
#cap = cv2.VideoCapture(0)

#create windows to display results
window_input, window_trackbar, window_output = jasf_cv.getBasicWindows()


def nothing(x):
    pass

#create track bars to stabilish color limits
mode = 'manual'
def onModeChange(opt):
    global mode
    mode = 'manual' if opt == 0 else 'mouse'
    print 'mode changed to', mode

jasf_cv.setTrackbar('manual/mouse mode', 0,1, onCallBack = onModeChange)


jasf_cv.setTrackbar('Hu', 50, 180)
jasf_cv.setTrackbar('Su', 50, 255)
jasf_cv.setTrackbar('Vu', 50, 255)

jasf_cv.setTrackbar('Hl', 0, 180)
jasf_cv.setTrackbar('Sl', 0, 255)
jasf_cv.setTrackbar('Vl', 0, 255)


jasf_cv.setTrackbar('delta H', 40, 100)
jasf_cv.setTrackbar('delta S', 40, 100)
jasf_cv.setTrackbar('delta V', 20, 100)


window_trackbar = 'settings'

def getHighColor():
    h = cv2.getTrackbarPos('Hu', window_trackbar)
    s = cv2.getTrackbarPos('Su', window_trackbar)
    v = cv2.getTrackbarPos('Vu', window_trackbar)
    return np.array([h,s,v])

def getLowColor():
    h = cv2.getTrackbarPos('Hl', window_trackbar)
    s = cv2.getTrackbarPos('Sl', window_trackbar)
    v = cv2.getTrackbarPos('Vl', window_trackbar)
    return np.array([h,s,v])

def getDeltaColor():
    h = cv2.getTrackbarPos('delta H', window_trackbar)
    s = cv2.getTrackbarPos('delta S', window_trackbar)
    v = cv2.getTrackbarPos('delta V', window_trackbar)
    return np.array([h,s,v])

def getColorSettings():
    return getHighColor(), getLowColor(), getDeltaColor()

def setMyTrackbarPos(color_center, delta):
    hu,su,vu = np.array(color_center) + np.array(delta)
    hl,sl,vl = np.array(color_center) - np.array(delta)

    cv2.setTrackbarPos('Hu', window_trackbar, hu)
    cv2.setTrackbarPos('Su', window_trackbar, su)
    cv2.setTrackbarPos('Vu', window_trackbar, vu)


    cv2.setTrackbarPos('Hl', window_trackbar, hl)
    cv2.setTrackbarPos('Sl', window_trackbar, sl)
    cv2.setTrackbarPos('Vl', window_trackbar, vl)

clicked_color = (0,0,0)
def getMousePositionColor(event,x,y,flags,param):
    global mode, hsv, clicked_color
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if mode == 'mouse':
            print 'mouse position updated to', np.array([x,y])

            delta = getDeltaColor()
            clicked_color = hsv[y, x, :]

            setMyTrackbarPos(clicked_color, delta) 


cv2.setMouseCallback(window_input, getMousePositionColor)

#initialize tracking colors
lower_color = getLowColor()
upper_color = getHighColor()

while (cap.isOpened()):
    ret, frame = cap.read()
    #convert color format to HSV to use later with color segmentation
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    if mode == 'mouse':
        delta = getDeltaColor()
        setMyTrackbarPos(clicked_color, delta) 

    lower_color, upper_color = getLowColor(), getHighColor()

    mask = cv2.inRange(hsv,lower_color,upper_color)
    
    cv2.imshow(window_input, frame)
    cv2.imshow(window_output, mask)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
       break
    
cap.release()
cv2.destroyAllWindows()
