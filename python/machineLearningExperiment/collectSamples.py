import sys
sys.path.insert(0, '../')
import cv2
import jasf
from jasf import jasf_cv
from devika import devika_cv

jasf_cv.getNewWindow('settings')
jasf_cv.getNewWindow('video')

import pathlib
videoFiles = list(p for p in pathlib.Path('../../video/mp4/').iterdir() if p.is_file() and p.name[0] != '.')

def printVideoFiles():
    print 'Here are the video files'
    for i,fn in enumerate(videoFiles):
        print i, fn.name
printVideoFiles()


cam = cv2.VideoCapture(videoFiles[0].absolute().as_posix())


#control variables
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

jasf_cv.setTrackbar('video file', 0, len(videoFiles)-1, onCallBack = onVideoChange)
jasf_cv.setTrackbar('pause', 0, 1)
jasf_cv.setTrackbar('positive', 0, 1)
jasf_cv.setTrackbar('width', 0, 50)
jasf_cv.setTrackbar('height', 0, 50)

input = 0
lastPositiveSaved = len(list(p for p in pathlib.Path('./samples/positive/').iterdir() if p.suffix == '.png'))
lastNegativeSaved = len(list(p for p in pathlib.Path('./samples/negative/').iterdir() if p.suffix == '.png'))

def readSettings():
    settings = dict()
    settings['w'] = jasf.cv.readTrackbar('width')
    settings['h'] = jasf.cv.readTrackbar('height')
    settings['positive'] = True if jasf.cv.readTrackbar('positive') == 1 else 0

    return settings

def onMouseDblClk(event, x,y, flags, param):
    """a callback function to capture the user input and add a template to the corner
    detectors """
    global input, lastNegativeSaved, lastPositiveSaved

    if event == cv2.EVENT_LBUTTONDBLCLK and (control_mode == 'pause'):
        winSize = 30

        #set square around clicked point
        top_left_x = max(0, x-winSize)
        top_left_y = max(0, y-winSize)

        bottom_right_x = min(input.shape[1], x+winSize)
        bottom_right_y = min(input.shape[0], y+winSize)

        #get the square from frame(input colored image)
        tpl = input[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        #set name of file to save
        settings = readSettings()
        name = ''
        if settings['positive']:
            name = './samples/positive/' +  str(lastPositiveSaved) +  '.png'
            lastPositiveSaved += 1
        else:
            name = './samples/negative/' +  str(lastNegativeSaved) + '.png'
            lastNegativeSaved += 1
        
        #save selected template to corresponding directory
        print 'printing new template to', name, '...'
        cv2.imwrite(name, tpl)

        #show selected template on screen
        img2show = input.copy()
        cv2.rectangle(img2show,(x-winSize, y-winSize), (x+winSize, y+winSize), (jasf.cv.red), 2)
        cv2.imshow('video', img2show)

cv2.setMouseCallback('video', onMouseDblClk)


while control_mode != 'end':
    control_mode = 'pause' if jasf.cv.readTrackbar('pause') == 1 else 'run'
    waitTime = 5 if control_mode == 'run' else 50
    ch = cv2.waitKey(waitTime) & 0xFF

    if ch == ord('p'):
        jasf.cv.switchBinnaryTrackbar('pause')

    if ch == ord('h'):
        printVideoFiles()
        
    if ch == ord('q'):
        print 'finishing due to user input'
        break

    if control_mode == 'pause':
        continue

    ret,frame = cam.read()
    input = frame

    if ret == False:
        print 'no frame coming out of of video stream'
        ch = cv2.waitKey(500) & 0xFF
        if ch == ord('q'):
            print 'finishing due to user input'
            control_mode = 'end'
        else:
            continue


    cv2.imshow('video', frame)


cv2.destroyAllWindows()
cam.release()

