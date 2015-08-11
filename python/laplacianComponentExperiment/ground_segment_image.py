help =  "\n\n HELP \n\n will run ground segmenting using shinny detection on one image \n\n"
print help
#20 mean shift algorithm
import sys
sys.path.insert(0, '../')
import numpy as np
import cv2

# break up an image into left and right
def break_left_right(img):
  h, w = img.shape[:2]
  left = img[:,1:w/2-1]
  right = img[:,w/2:w]
  return left,right
    
original = cv2.imread('../../img/sample.jpeg')
hsv = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
o_h, o_s, o_v = cv2.split(hsv)

window_trackbar = 'settings'
window_original = 'original'

cv2.namedWindow(window_original, cv2.WINDOW_NORMAL)


#window for trackbars
cv2.namedWindow(window_trackbar, cv2.WINDOW_NORMAL)
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.namedWindow('s', cv2.WINDOW_NORMAL)
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.namedWindow('sub', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)

cv2.moveWindow('original', 0, 0)
cv2.moveWindow('s', 400, 0)
cv2.moveWindow('dst', 800, 0)
cv2.moveWindow('sub', 0, 400)
cv2.moveWindow('mask', 400, 400)
cv2.moveWindow('settings', 800, 400)

# we are going to track the mostly white rat
def nothing(x):
    pass

#create track bars to stabilish color limits
cv2.createTrackbar('window_size',window_trackbar,3,13,nothing)
cv2.createTrackbar('th_min',window_trackbar,12,255,nothing)
cv2.createTrackbar('th_max',window_trackbar,219,255,nothing)
cv2.createTrackbar('erode',window_trackbar,3,13,nothing)
cv2.createTrackbar('dilate',window_trackbar,2,13,nothing)
cv2.createTrackbar('alpha approx',window_trackbar,16,50,nothing)
trackbar_order = '0:erode/dilate\n 1:dilate/erode'
cv2.createTrackbar(trackbar_order,window_trackbar,0,1,nothing)
cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))


while True:
    #read parameters from trackbars
    window_size = max(3,cv2.getTrackbarPos('window_size', window_trackbar))
    th_min = max(1, cv2.getTrackbarPos('th_min', window_trackbar))
    th_max = max(1, cv2.getTrackbarPos('th_max', window_trackbar))
    erode_p = cv2.getTrackbarPos('erode', window_trackbar)
    dilate_p = cv2.getTrackbarPos('dilate', window_trackbar)
    alpha = cv2.getTrackbarPos('alpha approx', window_trackbar)/1000.0
    control_order = cv2.getTrackbarPos(trackbar_order, window_trackbar)


    #compute mean of window around point
    dst = cv2.blur(o_s, (window_size, window_size))
    #find difference from point to mean
    sub = abs(dst - o_s)
    #find points in the regions of interest
    mask = cv2.inRange(sub,th_min,th_max)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleaning_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleaning_kernel)
    #some filtering
    if control_order == 0:
        mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),iterations = erode_p)
        mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),iterations = dilate_p)
    if control_order == 1:
        mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),iterations = dilate_p)
        mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),iterations = erode_p)

    target = mask.copy()

    ret, cnts, hier = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #find approximations to degree determined by epsilon
    approx = [cv2.approxPolyDP(c, alpha*cv2.arcLength(c,True), True) for c in cnts]

    #the floor is the one with the highest value of area
    roi = approx[np.argmax([cv2.contourArea(c) for c in approx])]


    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = cv2.drawContours(mask, cnts, -1, (0,0,255), 2)
    mask = cv2.drawContours(mask, [roi], -1, (255,0,0), 2)


    cv2.imshow('original', original)
    cv2.imshow('s', o_s )
    cv2.imshow('mask', mask)
    cv2.imshow('dst', dst)
    cv2.imshow('sub', sub)

    k = cv2.waitKey(5) & 0xff
    if k == ord('q'):
       break
    
cv2.destroyAllWindows()
