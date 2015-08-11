import sys
sys.path.insert(0, '../')
import cv2
import jasf
from jasf import jasf_cv
from devika import devika_cv
from multiFiltersDataStructures import *
import pathlib
import pickle
from filterExperimentUtil import *

#####################################
#DEFINE CONTROL VARIABLES   
control_Path2VideoFiles = '../../video/mp4/'
control_colorComponents2Caputre = ['v', 's', 'b']
control_componentsToCapture = control_colorComponents2Caputre + ['canny']
#accept: ['G', 'B', 'R', 'H', 'S', 'V', 'canny','laplace']

#####################################
#####################################
#Read video file names
#####################################
videoFiles = list(p for p in pathlib.Path(control_Path2VideoFiles).iterdir() if p.is_file() and p.name[0] != '.')

def printVideoFiles():
    print 'Here are the video files'
    for i,fn in enumerate(videoFiles):
        print i, fn.name
printVideoFiles()


cam = cv2.VideoCapture(videoFiles[0].absolute().as_posix())
#####################################
#set windows
#####################################
jasf_cv.getSettingsWindow()
jasf_cv.getNewWindow('settings2')
jasf_cv.getNewWindow('video', dimension = (320,240))
jasf.cv.getManyWindows(control_componentsToCapture, dim = (60,80), n=(6,6))
jasf_cv.getNewWindow('range')

#control variables
control_mode = 'run'


#####################################
#function to control video index
#####################################
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

#####################################
#set trackbars
#####################################
jasf_cv.setTrackbar('video file', 0, len(videoFiles)-1, onCallBack = onVideoChange, window_name='settings2')
jasf_cv.setTrackbar('pause', 0, 1, window_name='settings2')
jasf_cv.setTrackbar('l/r', 0, 1,window_name = 'settings2')
jasf.cv.setManyTrackbars(['h_step', 'v_step'], [100,100], [500,500],windowName = 'settings2')

#create one trackbar for each bar(horizontal and vertical)
jasf.cv.setManyTrackbars(['horizontal'+str(i) for i in (1,2,3,4)], startList=[int(x*1000) for x in (0.2,0.4,0.6,0.8)], maxList=[1000 for _ in (1,2,3,4)])
jasf.cv.setManyTrackbars(['vertical'  +str(i) for i in (1,2,3,4)], startList=[int(x*1000) for x in (0.2,0.4,0.6,0.8)], maxList=[1000 for _ in (1,2,3,4)])

#####################################
#required variables
#####################################
input = 0
horizontalsColors = [(0,0,255),(0,0,255), (0,255,255),(0,255,255) ]
verticalColors = [(255,0,0),(255,0,0), (255,255,0),(255,255,0) ]

#####################################
#auxiliary functions
#####################################
control_select_RL = 'left'
def readSettings():
    """Function to read line positions"""
    global control_select_RL
    settings = dict()
    h1,h2,h3,h4 = jasf.cv.readManyTrackbars(['horizontal'+str(i) for i in (1,2,3,4)])
    v1,v2,v3,v4 = jasf.cv.readManyTrackbars(['vertical'+str(i) for i in (1,2,3,4)])

    control_select_RL = 'left' if jasf.cv.readTrackbar('l/r',i=2) == 0 else 'right'


    return h1,h2,h3,h4,v1,v2,v3,v4

def readSteps():
    """ read the steps settings """
    h_step, v_step = jasf.cv.readManyTrackbars(['h_step', 'v_step'],i=2)
    return h_step, v_step

def incrementStep(incr=10, axis='row'):
    """ incremente one of the axis step """
    h,v = readSteps()
    name = 'h_step' if axis == 'col' else 'v_step'
    newVal = h + incr if axis == 'col' else v + incr

    jasf.cv.setManyTrackbarsPos([name], [newVal], i=2) 

def onMouseDblClk(event, x,y, flags, param):
    """A call back fucntion that will move the current set of lines to a position around the mouse click. The steps are
    used to compute the pisition."""
    global control_select_RL
    if event == cv2.EVENT_LBUTTONDBLCLK and (control_mode == 'pause'):
        h_step, v_step = readSteps()
        y = int(y/240.0 * 1000.0)
        x = int(x/320.0 * 1000.0)
        h1,h2 = y-h_step, y+h_step
        v1,v2 = x-v_step, x+v_step
        if control_select_RL == 'left':
            jasf.cv.setManyTrackbarsPos(['horizontal1', 'horizontal2', 'vertical1', 'vertical2'], [h1,h2,v1,v2])
        else:
            jasf.cv.setManyTrackbarsPos(['horizontal3', 'horizontal4', 'vertical3', 'vertical4'], [h1,h2,v1,v2])

cv2.setMouseCallback('video', onMouseDblClk)

def help():
    """ prints the equivalence between file numbers and the file being read. Also prints key functionalities"""
    printVideoFiles()
    print 'Commands:'
    commands = {'p':'play/pause', 'r':'switch mouse mode to set left/right mouse',\
                'h':'print help', 'w':'increment step by 10', 's':'decrement step by 10'}
    for letter,help_text in commands.items():
        print letter, ':', help_text

help()
def analizeKeyCommand(ch):
    """ analyse key command. Code removed from the main loop to allow better clarity of the main code """
    if ch == ord('p'):
        jasf.cv.switchBinnaryTrackbar('pause',i=2)
    if ch == ord('r'):
        jasf.cv.switchBinnaryTrackbar('l/r',i=2)
    if ch == ord('h'):
        help()
    if ch == ord('s'):
        incrementStep(10, axis = 'col')
    if ch == ord('x'):
        incrementStep(-10, axis = 'col')
    if ch == ord('c'):
        incrementStep(10, axis = 'row')
    if ch == ord('z'):
        incrementStep(-10, axis = 'row')

#####################################
#Set of operations applied to a common input
#####################################
multiFilter = FilterDealer()
for i in 'b':
    multiFilter.addFilter(getGBR_componentOperation(i))
for i in 'sv':
    multiFilter.addFilter(getHSV_componentOperation(i))
multiFilter.addFilter(cannyOperation())
#####################################
#Operation Objects
#####################################
lineClassifier = classifyLinesOperation()
rangePainter = rangePainterOperation()
#####################################
#MultiInput Operation Objects
#####################################
linePainter = paint4Horizontals4VerticalsMultiInputOperation()
gray2bgrConverter = convertGray2BGR_MultiInputOperation()
downSampler = pyramidDownMultiInputOperation(n=1)

#set settings for multifilter
#for the bgr, hsv 6 components
multiFilterSettings = [dict() for i in range(len(control_colorComponents2Caputre))]
#for the canny component
multiFilterSettings.append(cannyOperation.defaultParamDict)

#for classifying lines to zero or one
multiImgClassifyLinesOperation = classifyLinesMultiInputOperation()

#####################################
#Initialize collection list
#####################################

videoFiles = list(p for p in pathlib.Path(control_Path2VideoFiles).iterdir() if p.is_file() and p.name[0] != '.')
collectionList = []
if 'collectionList.pickle' in [p.as_posix() for p in pathlib.Path('./').iterdir()]:
    collectionList = pickle.load(open('./collectionList.pickle', 'r'))
    reportOnCollections(collectionList, 'loaded')
else:
    collectionList = [FilterSampleCollection(name) for name in ('b', 's', 'v', 'canny')]
    reportOnCollections(collectionList, 'created')
##########################################################################
#MAIN LOOP
##########################################################################
if __name__ == '__main__':
    while control_mode != 'end':
        control_mode = 'pause' if jasf.cv.readTrackbar('pause',i=2) == 1 else 'run'

        waitTime = 5 if control_mode == 'run' else 50
        ch = cv2.waitKey(waitTime) & 0xFF

        analizeKeyCommand(ch)
        if ch == ord('q'):
            print 'finishing due to user input'
            break
        if control_mode == 'run':
            ret,frame = cam.read()
        #test for end of video
        if ret == False:
            print 'no frame coming out of of video stream'
            ch = cv2.waitKey(500) & 0xFF
            if ch == ord('q'):
                print 'finishing due to user input'
                control_mode = 'end'
            else:
                continue

        input = frame

        h1,h2,h3,h4,v1,v2,v3,v4 = readSettings()
        horizontals = [h1,h2,h3,h4]
        verticals = [v1,v2,v3,v4]

        #get many grayscale components as defined by multiFilter
        inputList_gray = multiFilter.apply(frame.copy(), multiFilterSettings)
        B,S,V,canny= inputList_gray_reduced = downSampler.apply(inputList_gray)
        #convert those grayscales to GBR because we'd like to paint them
        inputList_color = gray2bgrConverter.apply(inputList_gray_reduced)
        #append the raw image to the previous image
        inputList_color.append(frame)

        #set settings to painter
        painterSettings = {'h1':h1,'h2':h2,'h3':h3,'h4':h4,\
                           'v1':v1,'v2':v2,'v3':v3,'v4':v4,\
                           'h_colors': horizontalsColors, 'v_colors':verticalColors }
        #paint the line as informed by the user on every image
        #this was more a debugging stage, to make sure I was cropping the right lines
        inputList_paint = linePainter.apply(inputList_color, [painterSettings for i in range(len(inputList_color))])

        #classify the lines of the raw image. PainterSettings contains the same info required here, that is, all the lines info
        rowLabels, colLabels = lineClassifier.apply(frame, painterSettings)
        #now that we know every label, we send them to the range painter. This was also a debug visualization of the classification
        rangePainterSettings = {'rowLabels':rowLabels, 'colLabels': colLabels}
        rangeOut = rangePainter.apply(frame, rangePainterSettings)

        if ch == ord('g'):
            print 'Capturing Samples'
            classiferSettings = painterSettings
            classiferSettings['collectionList'] = collectionList
            toBeClassfied = inputList_gray_reduced
            multiImgClassifyLinesOperation.apply(toBeClassfied, classiferSettings)


        #Show the components with marks of the dividing lines
        jasf.cv.imshow(['b', 's', 'v', 'canny'], inputList_paint[:len(inputList_paint)-1])
        #this is the main window were we can click to select
        cv2.imshow('video', inputList_paint[-1])
        #show the classified image
        cv2.imshow('range', rangeOut)

    #####################################
    #Store samples to file
    #####################################
    fn = './collectionList.pickle'
    with open(fn, 'w') as f:
        pickle.dump(collectionList, f)

    reportOnCollections(collectionList, 'saved')
    #####################################
    #Destroy windows and finish execution
    #####################################
    cv2.destroyAllWindows()
    cam.release()
