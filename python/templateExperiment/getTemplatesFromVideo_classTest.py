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
from util import TemplateDetector
from util import MovingTemplateDetector
from util import CornerWasLostException
import pickle

templates = [] 

def initializeTemplateDictFromFile(file):
    global templates
    templates = loadTemplatesFromPickle(file)


def initializeTemplateClass():
    global templates
    templates = {}
    for TCB in ('T', 'B'):
        for LCR in ('L', 'C', 'R'):
            #templates[TCB + LCR] = TemplateDetector(TCB + LCR)
            templates[TCB + LCR] = MovingTemplateDetector(TCB + LCR)

def writeTemplates2Pickle(templates, file):
    with open(file, 'wb') as f:
        pickle.dump(templates, f)

def loadTemplatesFromPickle(file):
    with open(file, 'r') as f:
        t = pickle.load(f)
    return t

initializeTemplateClass()

templateColorMap = {'TL':(0,0,255), 'TC':(0,0,150), 'TR':(0,100,50),
                    'BL':(255,0,0), 'BC':(150,0,0), 'BR':(50,0,100)}


#choose which video to play, the last one will be played
cam = cv2.VideoCapture('../../video/mp4/selected/walkingInTheBack.mp4')
cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves2.mp4')
cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves.mp4')
cam = cv2.VideoCapture('../../video/mp4/myFavoriteVideo.mp4')

#variable we will need:
control_mode = 'run'
frame = 0
control_last_saved = {}
img = 0

def onMouseDblClk(event, x,y, flags, param):
    """a callback function to capture the user input and add a template to the corner
    detectors """
    global templates,  input

    if event == cv2.EVENT_LBUTTONDBLCLK and (control_mode == 'pause'):
        print ' ok'
        winSize = 30

        #set square around clicked point
        top_left_x = max(0, x-winSize)
        top_left_y = max(0, y-winSize)

        bottom_right_x = min(input.shape[1], x+winSize)
        bottom_right_y = min(input.shape[0], y+winSize)

        #get the square from frame(input colored image)
        tpl = input[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        #read selected parameters to create name of the current selection
        TCB, LCR = readSettings()

        i = len(templates[TCB+LCR].templates)

        name = '../img/templates/' + TCB + LCR + str(i) + '.png'
        
        #save selected template to corresponding directory
        print 'printing new template to', name, '...'
        cv2.imwrite(name, tpl)

        #add selected template do corresponding dict
        templates[TCB+LCR].addTemplate(tpl) 

        #show selected template on screen
        img2show = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img2show,(x-winSize, y-winSize), (x+winSize, y+winSize), templateColorMap[TCB+LCR], 2)
        cv2.imshow('output', img2show)

jasf_cv.getNewWindow('output')
jasf_cv.getNewWindow('settings')
cv2.namedWindow('output')
cv2.setMouseCallback('output', onMouseDblClk)

def doNothing(val):
    pass

TCB_track = jasf_cv.setTrackbar('Top/Bottom',  0, 1) 
LCR_track = jasf_cv.setTrackbar('Left/Center/Rght',  0, 2) 
jasf_cv.setTrackbar('th',  40, 500) 

TCB_map = {0:'T', 1:'B'}
LCR_map = {0:'L', 1:'C', 2:'R'}

TCB_map_inverse = {a[1]:a[0] for a in TCB_map.items()}
LCR_map_inverse = {a[1]:a[0] for a in LCR_map.items()}

def readSettings():
    TCB = TCB_map[cv2.getTrackbarPos(TCB_track, 'settings')]
    LCR = LCR_map[cv2.getTrackbarPos(LCR_track, 'settings')]

    return TCB, LCR

    

while cam.isOpened():
    ch = cv2.waitKey(5) & 0xFF

    if ch == ord('q'):
     print 'finishing due to user input'
     break

    if ch == ord('p'):
        control_mode = 'pause' if control_mode == 'run' else 'run'

    if control_mode == 'run':

        ret,frame = cam.read()

        th = cv2.getTrackbarPos('th', 'settings') * 1000


        if ret == False:
            print 'finishing due to end of video'
            break

        left, right = devika_cv.break_left_right(frame)

        input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        input = cv2.equalizeHist(input)
        input = cv2.blur(input, (5,5))
        output = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)

        # Apply template Matching
        try:
            for TCB in 'TB':
                for LCR in 'LCR':
                    name = TCB + LCR
                    matches = []
                    top_left, bottom_right = [], []
                    templates[name].setWorkingTh(th)
                    top_left, bottom_right, ret = templates[name].track(input)
                    if ret == True:
                        for i in range(len(top_left)):
                            cv2.rectangle(output,top_left[i], bottom_right[i], templateColorMap[name], 2)
        except CornerWasLostException as E:
            name = E.name
            cornerTB = name[0]
            cornerLCR = name[1]

            cornerTB_val = TCB_map_inverse[cornerTB]
            cornerLCR_val = LCR_map_inverse[cornerLCR]

            cv2.setTrackbarPos(TCB_track, 'settings', cornerTB_val)
            cv2.setTrackbarPos(LCR_track, 'settings', cornerLCR_val)

            print 'please select new corners', name
            control_mode = 'pause'
            templates[name].resetFailureCount()

        cv2.imshow('output', output)


writeTemplates2Pickle(templates, '../pickle/templates_class.pickle')
cv2.destroyAllWindows()
cam.release()
