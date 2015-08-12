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


def help():
    print 'usage:\npython getClearSeizurse.py path2Videos inputFile path2output'
    quit()

if len(sys.argv) != 4:
    help()

#read user input
control_Path2VideoFiles = sys.argv[1]
control_Path2Input = sys.argv[2]
control_Path2Output = sys.argv[3]

#get list of files in input directory
videoFiles = list(p for p in pathlib.Path(control_Path2VideoFiles).iterdir() if p.is_file())

inputData = []
with open(control_Path2Input, 'r') as f :
    inputData = f.readlines()
    inputData = [x.replace('\n', '') for x in inputData]

print 'here are the input commands', inputData

for request in inputData:
    #read and fix input from request. We fix it so numbers are read as numbers and the path to the video is correct
    currentVideoFile, mouseInSeizure, startFrame, endFrame = request.split()
    currentVideoFile = control_Path2VideoFiles + currentVideoFile
    startFrame, endFrame = int(startFrame), int(endFrame)
    durationInFrameCount = endFrame - startFrame
    frameCount = 0

    #open video and set it to initial frame
    cam = cv2.VideoCapture(currentVideoFile)
    cam.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

    #get a video writer 
    originalFPS = cam.get(cv2.CAP_PROP_FPS)
    height, width = jasf.cv.getVideoCaptureFrameHeightWidth(cam)
    outputName = control_Path2Output + currentVideoFile.replace('/', '.').split('.')[4] + '_s' + str(startFrame) + 'f_e' + str(endFrame) + 'f.avi'
    videoWriter = cv2.VideoWriter(outputName, cv2.VideoWriter_fourcc(*'XVID'), originalFPS, (width, height))

    print 'querry', currentVideoFile, mouseInSeizure, startFrame, endFrame
    print 'saving to file', outputName

    control_mode = 'run'
    returMode = 'continue'

    while frameCount < durationInFrameCount:
        ret,frame = cam.read()
        videoWriter.write(frame)
        frameCount += 1


    cam.release()
    videoWriter.release()
