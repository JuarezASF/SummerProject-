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

templates = [] 
def initializeTemplate():
    global templates
    for i in (1,2,3,4,5,6):
        templates.append(cv2.imread('../../img/edge' + str(i) + '.png',0))

def initializeTemplateDict():
    global templates
    templates = {}
    for TCB in ('T', 'B'):
        for LCR in ('L', 'C', 'R'):
            templates[TCB + LCR] = []

def initializeTemplateDictFromFile(file):
    global templates
    templates = loadTemplatesFromPickle(file)

import pickle
def writeTemplates2Pickle(templates, file):
    with open(file, 'wb') as f:
        pickle.dump(templates, f)

def loadTemplatesFromPickle(file):
    with open(file, 'r') as f:
        t = pickle.load(f)
    return t


            
#initializeTemplateDict()
initializeTemplateDictFromFile('../pickle/templates.pickle')

templateColorMap = {'TL':(0,0,255), 'TC':(0,0,150), 'TR':(0,100,50),
                    'BL':(255,0,0), 'BC':(150,0,0), 'BR':(50,0,100)}



cam = cv2.VideoCapture('../../video/mp4/myFavoriteVideo.mp4')
cam = cv2.VideoCapture('../../video/mp4/selected/walkingInTheBack.mp4')
cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves2.mp4')
cam = cv2.VideoCapture('../../video/mp4/selected/camereMoves.mp4')
#variable we will need:
control_mode = 'run'
frame = 0
control_last_saved = {}
img = 0

def onMouseDblClk(event, x,y, flags, param):
    global templates,  input

    if event == cv2.EVENT_LBUTTONDBLCLK and (control_mode == 'pause'):
        print ' ok'

        #set square around clicked point
        top_left_x = max(0, x-10)
        top_left_y = max(0, y-10)

        bottom_right_x = min(input.shape[1], x+10)
        bottom_right_y = min(input.shape[0], y+10)

        #get the square from frame(input colored image)
        tpl = input[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        #read selected parameters to create name of the current selection
        TCB, LCR = readSettings()
        i = len(templates[TCB+LCR])
        name = '../img/templates/' + TCB + LCR + str(i) + '.png'
        
        #save selected template to corresponding directory
        print 'printing new template to', name, '...'
        cv2.imwrite(name, tpl)

        #add selected template do corresponding dict
        templates[TCB+LCR].append(tpl) 

        #show selected template on screen
        img2show = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img2show,(x-10, y-10), (x+10, y+10), templateColorMap[TCB+LCR], 2)
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
        for TCB in 'TB':
            for LCR in 'LCR':
                name = TCB + LCR
                matches = []
                for tmp in templates[name]:
                    w, h = tmp.shape[::-1]
                    res = cv2.matchTemplate(input,tmp,cv2.TM_SQDIFF)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
                    if min_val < th:
                        top_left = min_loc
                        bottom_right = (top_left[0] + w, top_left[1] + h)

                        cv2.rectangle(output,top_left, bottom_right, templateColorMap[name], 2)

        cv2.imshow('output', output)


writeTemplates2Pickle(templates, '../pickle/templates.pickle')
cv2.destroyAllWindows()
cam.release()
