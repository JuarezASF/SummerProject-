import cv2
import numpy as np

# open the video file
cap = cv2.VideoCapture("../video/mp4/2014-07-16_08-41-11.mp4")
# read the first frame
ret,frame1 = cap.read()
# take that frame and convert to gray scale (this will be the previous image for optical flow)
prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

framecnt = 0
while (framecnt < 100):
    ret,frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    # compute dense optical flow -- track all features
    flow = cv2.calcOpticalFlowFarneback(prev,next, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prev = next
    framecnt += 1
    
cap.release()
cv2.destroyAllWindows()
    
        
