import cv2
import numpy as np

# open the video file
cap = cv2.VideoCapture("../video/mp4/2014-07-16_08-41-11.mp4")
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# read the first frame
ret,frame1 = cap.read()
# take that frame and convert to gray scale (this will be the previous image for optical flow)
prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame1)

# Create some random colors
color = np.random.randint(0,255,(100,3))

framecnt = 0
while (framecnt < 200):
    ret,frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev, next, p0, None, **lk_params)

     # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame2,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame2,mask)
    
    cv2.imshow('frame2',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        
    # now update the previous frame and previous points
    prev = next
    p0 = good_new.reshape(-1,1,2)
    framecnt += 1
    
cap.release()
cv2.destroyAllWindows()
    
        
