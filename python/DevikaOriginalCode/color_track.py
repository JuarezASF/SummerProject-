#20 mean shift algorithm
import numpy as np
import cv2
from jasf import jasf_cv

    
fn = "../video/mp4/2014-07-16_08-41-11.mp4"
cap = cv2.VideoCapture()
cap.open(fn)
#cap = cv2.VideoCapture(0)

#create windows to display results
window_input = jasf_cv.getNewWindow('input')
window_output = jasf_cv.getNewWindow('output')

#window for trackbars
window_trackbar = jasf_cv.getNewWindow('settings')


def nothing(x):
    pass

#create track bars to stabilish color limits
mode = 'manual'
def onModeChange(opt):
    mode = 'manual' if opt == 0 else 'mouse'
    print 'mode changed to', mode

cv2.createTrackbar('manual/mouse mode',window_trackbar,0,1,onModeChange)

cv2.createTrackbar('Hu',window_trackbar,0,180,nothing)
cv2.createTrackbar('Su',window_trackbar,0,255,nothing)
cv2.createTrackbar('Vu',window_trackbar,0,255,nothing)

cv2.createTrackbar('Hl',window_trackbar,0,180,nothing)
cv2.createTrackbar('Sl',window_trackbar,0,255,nothing)
cv2.createTrackbar('Vl',window_trackbar,0,255,nothing)

cv2.createTrackbar('delta H',window_trackbar,0,20,nothing)
cv2.createTrackbar('delta S',window_trackbar,0,20,nothing)
cv2.createTrackbar('delta V',window_trackbar,0,20,nothing)

cv2.setTrackbarPos('Hu', window_trackbar, 50)
cv2.setTrackbarPos('Su', window_trackbar, 50)
cv2.setTrackbarPos('Vu', window_trackbar, 50)

cv2.setTrackbarPos('Hl', window_trackbar, 0)
cv2.setTrackbarPos('Sl', window_trackbar, 0)
cv2.setTrackbarPos('Vl', window_trackbar, 0)

cv2.setTrackbarPos('delta H', window_trackbar, 5)
cv2.setTrackbarPos('delta S', window_trackbar, 5)
cv2.setTrackbarPos('delta V', window_trackbar, 5)

cv2.setTrackbarPos('manual/mouse mode', window_trackbar, 0)

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

mouse_x = 0
mouse_y = 0
mouse_pressed = False

def getMousePositionColor(event,x,y,flags,param):
    global mouse_x, mouse_y, mode
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if mode == 'mouse':
            mouse_x = x
            mouse_y = y
        print 'mouse position updated to', np.array([x,y])

cv2.setMouseCallback(window_input, getMousePositionColor)

#initialize tracking colors
lower_color = getLowColor()
upper_color = getHighColor()

while (cap.isOpened()):
    ret, frame = cap.read()
    #convert color format to HSV to use later with color segmentation
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    if mode == 'manual':
        lower_color = getLowColor()
        upper_color = getHighColor()
    if mode == 'mouse':
        delta = getDeltaColor()
        color_mouse = frame[mouse_x, mouse_y]
        lower_color = color_mouse - delta
        upper_color = color_mouse + delta

    mask = cv2.inRange(hsv,lower_color,upper_color)
    
    
    cv2.imshow(window_input, frame)
    cv2.imshow(window_output, mask)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
       break
    
cam.release()
cv2.destroyAllWindows()
