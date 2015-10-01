"""
Read all files in the input directory. If they are .avi files, write N files to the output direcoty,
each one containing at maximum 60 seconds of the original video. Files are names in the same way the original one,
extended with '_ptXX'
"""
import sys
import cv2
import pathlib
sys.path.append('../../python')
import jasf
from jasf import jasf_cv

path2Videos = "/home/jasf/Documents/Copy/RiceUniversity/Summer/Research/video/avi/"

videoFiles = [f for f in pathlib.Path(path2Videos).iterdir() if f.is_file() and f.name[0] != '.']

for f in videoFiles:
    fileName = f.as_posix()
    print 'name of file being processed: \n' , fileName

    #get file radical(we know it ends with .avi
    fileRadical = f.name[0:len(f.name)-len(f.suffix)]

    #open video and set it to initial frame
    cam = cv2.VideoCapture(fileName)
    if not cam.isOpened():
        print 'could not open file', fileName
        break

    #get original height and width
    height, width = jasf.cv.getVideoCaptureFrameHeightWidth(cam)

    #determine number of parts to split
    #we'll have one parte for every minute of video
    videoFrameCount = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    videoFrameRate = cam.get(cv2.CAP_PROP_FPS)
    minutusInVideo = videoFrameCount/(60.0*videoFrameRate)
    numberOfParts = int(minutusInVideo + 0.5)

    print 'file', fileName, ' would produce:'
    frames2read = videoFrameCount
    framesPerPart = int(60.0*videoFrameRate + 0.5)

    for currentParte in range(numberOfParts):

        currentOutputVideo = './output/%s_pt%02d.avi'%(fileRadical, currentParte)
        print currentOutputVideo

        #get a video writer 
        videoWriter = cv2.VideoWriter(currentOutputVideo, cv2.VideoWriter_fourcc(*'XVID'), videoFrameRate, (width, height))
        currentFrameRead = 0
        while (currentFrameRead < framesPerPart) and (frames2read > 0) and cam.isOpened() :
            ret,frame = cam.read()
            if ret == False:
                print 'failed to read frame from', fileRadical
            videoWriter.write(frame)
            currentFrameRead += 1
            frames2read -= 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print 'ending this process due to user input'
                break

        print 'releasing video', currentOutputVideo
        videoWriter.release()

        print 'releasing input video', fileName
    cam.release()
