import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from cv2 import imshow
import jasf
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
import pathlib

control_Path2VideoFiles = '../../video/mp4/'
videoFiles = list(p for p in pathlib.Path(control_Path2VideoFiles).iterdir() if p.is_file() and p.name[0] != '.')
cam = cv2.VideoCapture(videoFiles[0].absolute().as_posix())
control_mode = 'run'
def onVideoChange(index):
    """Function to be called when video trackbar is moved. It will re initializate the
    global variable cam"""
    global cam, control_mode
    control_mode = 'pause'
    cam.release()
    fn = videoFiles[index].absolute().as_posix() 
    print 'opening', fn
    cam = cv2.VideoCapture(fn)
    control_mode = 'run'

def onVideoPositionChange(index, cam):
    control_mode = 'pause'
    cam.set(cv2.CAP_PROP_POS_FRAMES, index)
    control_mode = 'run'

jasf_cv.getNewWindow('input')
jasf_cv.getNewWindow('output')
jasf_cv.getNewWindow('settings')
#####################################
#set trackbars
#####################################
jasf_cv.setTrackbar('video file', 0, len(videoFiles)-1, onCallBack = onVideoChange, window_name='settings')
jasf_cv.setTrackbar('pos', 0, int(cam.get(cv2.CAP_PROP_FRAME_COUNT)), onCallBack = lambda i: onVideoPositionChange(i, cam))

counter = 0
control_FPS = int(cam.get(cv2.CAP_PROP_FPS))
while cam.isOpened():
    waitTime = 5 if control_mode == 'run' else 50
    ch = cv2.waitKey(waitTime) & 0xFF
    if ch == ord('q'):
        print 'finishing due to user input'
        break
    if ch == ord('p'):
        control_mode = 'run' if control_mode == 'pause' else 'pause'
    if control_mode == 'run':
        ret,frame = cam.read()
        counter += 1
        if counter == control_FPS:
            jasf.cv.setTrackbarPos('pos', int(cam.get(cv2.CAP_PROP_POS_FRAMES)))
            counter = 0
    if ret == False:
        control_mode = 'pause'
        continue

    cv2.imshow('input', frame)

cv2.destroyAllWindows()
cam.release()
