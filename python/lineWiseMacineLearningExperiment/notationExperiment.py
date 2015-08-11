import cv2
import numpy as np

img = cv2.imread('../../img/sample.jpeg', 1)
h,w,d = img.shape

img[0:10, 0:w] = (0,0,0)

img[0:h, 0:10] = (0,0,255)


cv2.imshow('out', img)

cv2.waitKey(0)

