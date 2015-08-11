# crop the image into left and right for the two mice
# check out histogram equalization
import numpy as np
import cv2

# break up an image into left and right
def break_left_right(img):
  h, w = img.shape[:2]
  left = img[:,1:w/2-1]
  right = img[:,w/2:w]
  return left,right
    
cam = cv2.VideoCapture("../video/mp4/2014-07-16_08-41-11.mp4")
ret, frame = cam.read()
left,right = break_left_right(frame)
right_gray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
#clahe = cv2.createCLAHE()
#right_gray_eq = clahe.apply(right_gray)
fgbg = cv2.BackgroundSubtractorMOG(50,5,0.6,20)
fgmask = fgbg.apply(right_gray)

framecnt = 0
while (framecnt < 100):
  ret, frame = cam.read()
  left,right = break_left_right(frame)
  right_gray = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
  #right_gray_eq =clahe.apply(right_gray)
  fgmask = fgbg.apply(right_gray,fgmask,0.01)
  
  
  # do contour building on gray scale
  ret,thresh = cv2.threshold(right_gray,150,170,0)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  areas = [cv2.contourArea(contours[i]) for i in range(len(contours))]
  sareas = sorted(areas)
  ind = areas.index(sareas[-3])
  cv2.drawContours(right, contours, ind, (0,255,0), 3)
  
  cv2.imshow('frame',fgmask)
  cv2.imshow('gray',right)
  
  k = cv2.waitKey(30) & 0xff
  if k == 27:
      break
  framecnt += 1
cam.release()
cv2.destroyAllWindows()
