import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from cv2 import imshow
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from util import DetectMovement

from config import video2load
cam = cv2.VideoCapture('../video/mp4/selected/camereMoves.mp4')
cam = cv2.VideoCapture('../video/mp4/selected/notMoving.mp4')
cam = cv2.VideoCapture('../video/mp4/selected/walkingInTheBack.mp4')
cam = cv2.VideoCapture('../video/mp4/selected/camereMoves2.mp4')
cam = cv2.VideoCapture('../video/avi/myFavoriteVideo.avi')
cam = cv2.VideoCapture(video2load)

jasf_cv.getNewWindow('input')
jasf_cv.getNewWindow('output')
jasf_cv.getNewWindow('settings')

jasf_cv.setTrackbar('movingTh', 2, 10)
jasf_cv.setTrackbar('warningTh', 15, 100)
jasf_cv.setTrackbar('dangerTh', 85, 100)
jasf_cv.setTrackbar('waitTime', 200, 360)
jasf_cv.setTrackbar('sx', 40, 100)
jasf_cv.setTrackbar('sy', 40, 100)

def readSettings():
    th1 = cv2.getTrackbarPos('movingTh', 'settings')
    th2 = cv2.getTrackbarPos('warningTh', 'settings')*0.01
    th3 = cv2.getTrackbarPos('dangerTh', 'settings')*0.01
    time = cv2.getTrackbarPos('waitTime', 'settings')
    sx = max(1,cv2.getTrackbarPos('sx', 'settings'))
    sy = max(1,cv2.getTrackbarPos('sy', 'settings'))

    return th1, th2,th3,time, sx, sy


ret,frame = cam.read()
if ret == False:
    print 'finishing due to end of video'
    quit()
gray_initial = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5))

detector = DetectMovement(gray_initial)


def addVal2Component(img, color):
    mask = np.full_like(img, color)
    return img + mask

while cam.isOpened():
    ret,frame = cam.read()
    if ret == False:
        print 'finishing due to end of video'
        break
    frame = cv2.blur(frame, (5,5))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    movingTh, warningTh, dangerTh, waitTime, sx, sy = readSettings()

    detector.setMovingTh(movingTh)
    detector.setWarningTh(warningTh)
    detector.setDangerTh(dangerTh)
    detector.setWaitTime(waitTime)
    detector.setSteps(sx, sy)

    displacements,r = detector.detect(gray)

    out = gray.copy()
    out = jasf_cv.convertGray2BGR(out)


    #draw circles 
    for  line in displacements:
        p0 = line[0]
        p1 = line[1]
        if r == DetectMovement.globalMovingStatus_normal:
            cv2.circle(out, tuple(p0), 1, (0, 255, 0), -1)
        elif r == DetectMovement.globalMovingStatus_camera_move:
            cv2.circle(out, tuple(p0), 1, (255, 0, 0), -1)
            out = addVal2Component(out, (0,0,150))
        elif r == DetectMovement.globalMovingStatus_backMovement:
            cv2.circle(out, tuple(p0), 1, (0, 0, 255), -1)
            out = addVal2Component(out, (155,0,0))

    cv2.imshow('input', gray)
    cv2.imshow('output', out)
    
    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
     print 'finishing due to user input'
     break


cv2.destroyAllWindows()
cam.release()


