import sys
sys.path.insert(0, '../')
import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
from cv2 import imshow
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv

img = cv2.imread('../../img/sample.jpeg',0)
img2 = img.copy()
template = [] 
for i in (1,2,3,4,5,6):
    template.append(cv2.imread('../../img/edge' + str(i) + '.png',0))

w, h = template[0].shape[::-1]


cam = cv2.VideoCapture('../../video/mp4/myFavoriteVideo.mp4')

while cam.isOpened():
    ret,frame = cam.read()
    if ret == False:
        print 'finishing due to end of video'
        break
    left, right = devika_cv.break_left_right(frame)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply template Matching
    for tmp in template:
        res = cv2.matchTemplate(img,tmp,cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)

    cv2.imshow('output', img)

    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
     print 'finishing due to user input'
     break

cv2.destroyAllWindows()
cam.release()
