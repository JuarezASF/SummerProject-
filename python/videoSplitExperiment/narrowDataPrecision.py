""" This program will help narrow down the precision of the data we have. It will show two windows, one with the video
and one with a selection of the video that plays over and over. Use the trackbars on the settings window to find the
precise selection around the seizure moment. Press 'g'(if I am not mistaken) to ge the data. Remember to set which mouse
is having the seizure."""


import sys
sys.path.insert(0, '../')
import numpy as np
import cv2
from cv2 import imshow
from jasf import jasf_cv
jasf_cv.switchSilentMode()
from jasf import jasf_ratFinder
from devika import devika_cv
import pathlib
import jasf
from datetime import datetime 

control_Path2VideoFiles = './output/presentationVideo/'
videoFiles = list(p for p in pathlib.Path(control_Path2VideoFiles).iterdir() if p.is_file())
#videos will be presented in choronological order. Here is how the wonders of regular expressions work
videoFiles.sort(key = lambda x: x.name)


def readSettings():
    return jasf.cv.readManyTrackbars(['pos', 'start', 'end'])

report = open('./log.txt', 'a')
report.write('\nprogram run at %s \n'%datetime.now())

for file in videoFiles:
    jasf_cv.resetIndexes()
    jasf_cv.getNewWindow('input')
    jasf_cv.getNewWindow('output')
    jasf_cv.getNewWindow('settings')
    cam = cv2.VideoCapture(file.as_posix())
    cam2 = cv2.VideoCapture(file.as_posix())

    frameCount = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    control_mode = 'run'
    returMode = 'continue'

    count = 0
    control_uptdatePos = 48
    currentSelectionIndex = 0


    def movedPosSlided(index, cam):
        global control_mode
        oldControl = control_mode
        control_mode = 'pause'
        cam.set(cv2.CAP_PROP_POS_FRAMES, index)
        control_mode = oldControl

    jasf_cv.setTrackbar('pos', 0, frameCount, onCallBack = lambda i: movedPosSlided(i, cam))
    jasf_cv.setTrackbar('start', 0, frameCount)
    jasf_cv.setTrackbar('end', frameCount, frameCount)
    jasf_cv.setTrackbar('l/r', 0, 1)

    def switchLR():
        LR = jasf.cv.readTrackbar('l/r')
        LR = 1 if LR == 0 else 0
        jasf.cv.setTrackbarPos('l/r', LR)
    
    pos, start, end = -1, -1, -1

    while True:
        waitTime = 50  if control_mode == 'run' else 50
        ch = cv2.waitKey(waitTime) & 0xFF
        if ch == ord('q'):
            print 'finishing due to user input'
            break
        if ch == 27:
            returMode = 'end'
            break
        if ch == ord('p'):
            control_mode = 'run' if control_mode == 'pause' else 'pause'
        if ch == ord('r'):
            switchLR()
        if ch == ord('g'):
            lr = 'righ' if jasf.cv.readTrackbar('l/r') == 1 else 'left'
            report.write('video ' + file.name + ' ' + lr + ' mouse seizure from frame %05d'%start + ' to frame %05d'%end + '\n')

        if control_mode == 'run':
            pos, start, end = readSettings()

            ret,frame = cam.read()
            count += 1
            if count == control_uptdatePos:
                count = 0
                jasf.cv.setTrackbarPos('pos', jasf.cv.readTrackbar('pos') + control_uptdatePos)

            if ret:
                cv2.imshow('input', frame)
            else:
                control_mode = 'pause'

            cam2.set(cv2.CAP_PROP_POS_FRAMES, start + currentSelectionIndex)
            ret,selectionFrame = cam2.read()
            currentSelectionIndex = currentSelectionIndex + 1
            if currentSelectionIndex > end - start:
                currentSelectionIndex = 0

            if ret:
                cv2.imshow('output', selectionFrame)



    cv2.destroyAllWindows()
    cam.release()
    report.write('\n')

    if returMode == 'end':
        print 'finishing global execution'
        break

report.close()
