import sys
sys.path.insert(0, '../')
import cv2
import numpy as np
from config import video2load
cap = cv2.VideoCapture(video2load)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

noiteFilterKernelSize = (7,7)

while(1):
    for i in range(5):
        ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    next = cv2.blur(next, noiteFilterKernelSize)

    flow = cv2.calcOpticalFlowFarneback(\
        prev=prvs, next=next, flow=None,\
        pyr_scale = 0.5, levels=3, winsize=15,\
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    _,mag = cv2.threshold(src=mag, thresh=1.0, maxval=0.1, type=cv2.THRESH_TOZERO)
    #hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()
