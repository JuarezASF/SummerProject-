# mean shift algorithm
import numpy as np
import cv2

# break up an image into left and right
def break_left_right(img):
  h, w = img.shape[:2]
  left = img[:,1:w/2-1]
  right = img[:,w/2:w]
  return left,right
    
cam = cv2.VideoCapture("../video/mp4/2014-07-16_08-41-11.mp4")
# take first frame of video
ret, frame = cam.read()
left,right = break_left_right(frame)

# set up initial location of window

track_window = (0,125,50,40)


x,y,w,h = track_window
cv2.rectangle(right,(x,y),(x+w,y+h),255,2)
cv2.imshow('rect',right)
 
roi = right[y:y+h,x:x+w,:] 


hsv = cv2.cvtColor(right,cv2.COLOR_BGR2HSV)
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
hsv_ref_roi = hsv_roi
mask = cv2.inRange(hsv_roi, np.array((0., 0.,100.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0,1],mask,[180],[0,180])
roi_hist = cv2.calcHist([hsv_roi],[0,1],None,[180,255],[0,180,0,255])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,180,0,255],1)

# preserve this first histogram to reacquire the rat if we lose it during mean shift
ref_roi_hist = roi_hist

# convolve with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
cv2.filter2D(dst,-1,disc,dst)

# threshold and binary AND
ret,thresh = cv2.threshold(dst,100,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(right,thresh)

res = np.vstack((right,thresh,res))
#cv2.imshow('res',res)

# set up termination criteria
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1 )

prevx, prevy, prevw, prevh = x,y,w,h

framecnt = 0
while (framecnt < 1050):
    ret, frame = cam.read()
    
    left,right = break_left_right(frame)
    x,y,w,h = track_window
    roi = right[y:y+h,x:x+w,:]
    
    hsv = cv2.cvtColor(right,cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    
    roi_hist = cv2.calcHist([hsv_roi],[0,1],None,[180,255],[0,180,0,255])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsv],[0,1],roi_hist,[0,180,0,255],1)
    
    # clean out dst using morphological operations
    # remove small dots
    # suppress regions far from current window
    kernel = np.ones((3,3),np.uint8)
    dst = cv2.morphologyEx(dst,cv2.MORPH_OPEN,kernel)
    cv2.imshow('erosion',dst)
    
    #disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    #cv2.filter2D(dst,-1,disc,dst)

    
    ret,thresh = cv2.threshold(dst,200,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(right,thresh)

    res = np.vstack((right,thresh,res))
    cv2.imshow('res',res)
    
    # what is the distance between roi_hist and ref_roi_hist?
    #ref_dist = cv2.compareHist(roi_hist,ref_roi_hist,cv2.cv.CV_COMP_CORREL)
    abs_dist = abs(prevx-track_window[0])+abs(prevy-track_window[1])
    
    if abs_dist > 25:
       print "Lost rat at framecnt ", framecnt, abs_dist
       print "previous track window = ",prevx, prevy, prevw, prevh
       print "new track-window = ",track_window
       
       print "Reacquiring rat using template matching..."
       res = cv2.matchTemplate(hsv,hsv_ref_roi,cv2.TM_SQDIFF)
       min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
       top_left = min_loc
       bottom_right = (top_left[0] + w, top_left[1] + h)

       cv2.rectangle(right,top_left, bottom_right, 255, 2)
       cv2.imshow('res',right)
       
       

      
    prevx, prevy, prevw, prevh = track_window
    # apply meanshift to get new location
    ret,track_window = cv2.meanShift(dst,track_window,term_crit)
    print "track-window = ",track_window, "abs dist = ",abs(prevx-track_window[0])+abs(prevy-track_window[1])
    # draw it on the image
    x,y,w,h = track_window
    cv2.rectangle(right,(x,y),(x+w,y+h),255,2)
    cv2.imshow('rect',right)
    
    k = cv2.waitKey(60) & 0xff
    if k == 27:
       break
    
    framecnt = framecnt + 1
    
cam.release()
#cv2.destroyAllWindows()
