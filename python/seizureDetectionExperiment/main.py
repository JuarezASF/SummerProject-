import pdb
import numpy as np
import cv2
from cv2 import imshow
import pathlib
import sys
sys.path.insert(0, '../')
sys.path.append('../devikaFilterExperiment/')
import jasf
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
from DevikasFilterGroundSubtraction_ContourFinder import DevikasFilterGroundSubtraction_ContourFinder
from PreviousCenter_MousePicker import PreviousCenter_MousePicker
import copy
sys.path.append('../flowExperiment')
from flowUtil import FlowComputer
import flowUtil
import rangePercentFlowComputer
import windowFlowUtil
import json

def help():
    print 'usage: python main.py option=choice'
    print 'options:'
    print 'flowComputer: regular, rangePercent_regular, rangePercent_windowFlow, windowFlow'

#videos will be read from this path
control_Path2VideoFiles = '../videoCleanExperiment/output/fallingMouse/'
control_Path2VideoFiles = '../../video/mp4/'
control_settings = {'control_mode':'run', 'currentVideoFileName':'NONE'}
control_inputDict = jasf.parseInput(sys.argv)

videoFiles = list(p for p in pathlib.Path(control_Path2VideoFiles).iterdir() if p.is_file() and p.name[0] != '.')
videoFiles.sort(key = lambda x: x.name)
print videoFiles
cam = cv2.VideoCapture(videoFiles[0].absolute().as_posix())

jasf_cv.getNewWindow('settings')
jasf_cv.getNewWindow('settings1')
jasf.cv.getManyWindows(['B', 'otsuTh', 'clean+filter', 'tracking', 'flowImg'], n = (5,5))
#####################################
#set trackbars
#####################################
def onVideoChange(index):
    """Function to be called when video trackbar is moved. It will re initializate the
    global variable cam"""
    global cam
    setControlSetting('control_mode', 'pause')
    cam.release()
    fn = videoFiles[index].absolute().as_posix() 
    print 'opening', fn
    cam = cv2.VideoCapture(fn)

    #get the filename from the index read. The name stores is just the filename, no path included
    fn = videoFiles[index].name
    setControlSetting('currentVideoFileName', fn)

    #control_settings['currentVideoFileName'] = fn
    if fn not in userInputData.keys():
        #add video name entry to dictionary of user inputs
        print 'adding video', fn, ' to list os known videos'
        userInputData[fn] = list()
        setControlSetting('listOfKnownFrames_with_input', [])
    else:
        #read userInput data and store the frame numbers with available data
        framesWithInput = [x['frame'] for x in userInputData[fn]]
        setControlSetting('listOfKnownFrames_with_input', framesWithInput)
        print 'list of frames with input for file', fn, ':\n', framesWithInput

    cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cam.read()
    cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    setControlSetting('control_mode', 'run')
    setControlSetting('framesSinceStart', 0)

    if 0 in control_settings['listOfKnownFrames_with_input']:
        #read current video file name
        currentVideoFileName_aux = control_settings['currentVideoFileName']
        #find in the user input the one corresponding to the current frame; the returned type is a list
        userInputForThisFrame = filter(lambda x: x['frame'] == control_settings['framesSinceStart'], userInputData[currentVideoFileName_aux])
        askUserForInput(frame, True, userInputForThisFrame[0])
        #print a silly msg so we know something was done
        print 'aha! we avoided that!'
    else:
        askUserForInput(frame)

    resetPlotData()



jasf_cv.setTrackbar('video file', 0, len(videoFiles)-1, onCallBack = onVideoChange, window_name='settings')

#first load a dictionary with the current configuration and then set all trackbars at once
paramDict = dict()
with open('./paramDict.json', 'r') as f:
    paramDict = json.load(f)

def onWindowFlowSizeChanged(val):
    global flowComputerOption, flowComputer
    if 'windowFlow' in flowComputerOption:
        print 'changing window size for flow averaging to', val
        flowComputer.setWindowSize(max(val,1))

for item in paramDict.items():
    itemName = item[0]
    itemVal = item[1]
    if 'onCallBack' in itemVal.keys():
        jasf_cv.setTrackbar(itemName, itemVal['currentVal'], itemVal['max'], i = itemVal['window'], onCallBack=locals()[itemVal['onCallBack']])
    else:
        jasf_cv.setTrackbar(itemName, itemVal['currentVal'], itemVal['max'], i = itemVal['window'])

"""
userInputData is a dictionary where the keys are names of videos and the item is a list of dicts. Each element of this
list contain three items: 'frame', 'input' and 'settings_state'.
frame: an integer indicating the frame index of the input
input: a tuple containing the x and y coordinate of the input
settings_state: a dict containing the settings when the user clicked on the mouse

The idea is that this data structure contains everything to recreate the user input.

the userInputData is used on the following moments:
    *when the user clicks on the mouse, a new entry is added to the dictionary
    *when the user changes the video the dictionary is read for that entry; if there is no entry for the current video,
    then a new one is created

The program must keep track of the current frame number, when the frame number is equal the one where the input happend,
the program will set all the settings and the mouse location for that indicated by the data.


How to use it:
    1. the current video filename is stored into control_settings['currentVideoFileName']
    2. the current frames with input are stored into control_settings['listOfKnownFrames_with_input']
    3. read the current input data with:
        userInputData[control_settings['currentVideoFileName']]
"""
userInputData = dict()
userInputData_file = './userInputData.json'
if pathlib.Path(userInputData_file).is_file():
    print 'loading user input from', userInputData_file
    with open(userInputData_file, 'r') as f:
        userInputData = json.load(f)
else:
    print 'initializing user inpur data as empty dictionary'

#####################################
#Auxiliar Classes
#####################################
class MouseDescription:
    def __init__(self):
        self.rx = -1
        self.ry = -1
        self.mouse = []

    def setPosition(self, rx, ry, mouse = []):
        self.rx = rx
        self.ry = ry 
        if len(mouse) > 0:
            self.mouse = copy.deepcopy(mouse)

#####################################
#Control Variables
#####################################
control_mouse = MouseDescription()
control_show_plot = True
control_show_fft_fft = False
control_array = {}
control_array['fps'] = int(cam.get(cv2.CAP_PROP_FPS) + 0.5)
control_array['frames2FFT'] = 3*control_array['fps']
data2FFT = []
control_array['framesSinceStart'] = 0
control_array['listOfKnownFrames_with_input'] = []
#####################################
#Auxiliar Functions
#####################################

def setControlSetting(name, val):
    global control_settings
    control_settings[name] = val

def readControlSetting(name):
    global control_settings
    return control_settings[name]

def readSettingsState():
    global paramDict
    for item in paramDict.items():
        itemName, itemVal = item[0], item[1]
        paramDict[itemName]['currentVal'] = jasf.cv.readTrackbar(itemName, i=itemVal['window'])

    return copy.deepcopy(paramDict)

def setSettingsState(paramDict):
    for item in paramDict.items():
        itemName, itemVal = item[0], item[1]
        jasf.cv.setTrackbarPos(itemName, itemVal['currentVal'], i=itemVal['window'])

def readSettings():
    """ read general settings """
    return jasf.cv.readManyTrackbars(['th', 'max', 'delta', 'dilateSize', 'erodeSize', 'LRA'])

def readRangePercentSettings():
    return jasf.cv.readManyTrackbars(['rangePercent_min', 'rangePercent_max' ], i=1)

def readWindowFlowSize():
    return jasf.cv.readTrackbar('windowFlow_size', i=1)

def readFlowSettings():
    """ read flow settings """
    thSet = jasf.cv.readManyTrackbars(['flow_lowTh', 'flow_upTh', 'flowConect_lowTh', 'flowConect_upTh'])
    onSet = [bool(v) for v in jasf.cv.readManyTrackbars(['connectivityFilterOn', 'magnitudeFilterOn'], i=1)]
    return thSet+onSet

def switchRunPause():
    """ switch the 'control_mode' trackbar on the 'settings' window"""
    new_mode = 'run' if readControlSetting('control_mode') == 'pause' else 'pause'
    setControlSetting('control_mode', new_mode)

def updateValuesOfTh(area_delta):
    """update the values of threshold based on the area of the current self.mouse. The new information is passed to the
    rest of the code by setting the trackbars to the new values. The program will later read the trackbars and that's
    how this function communicate with the rest."""
    #compute area and set the new threshoulds
    area = int(cv2.contourArea(control_mouse.mouse)/100)
    newTh = max(area - area_delta, 0)
    newTh_max = max(area + area_delta, 16)
    jasf.cv.setManyTrackbarsPos(['th', 'max'], [newTh, newTh_max])

def askUserForInput(frame, modeReadingFromData=False, userData=dict()):
    """Function will read the settings, find interesting contours and show then to the user so he can pick the correct
    contour """
    global control_mouse
    #read image and break into right and left
    left, right = devika_cv.break_left_right(frame)
    inputList = [left, right, frame]

    #create window to wait input
    jasf_cv.getNewWindow('user input', dimension=(160,120))

    cnts = []
    control_mouse.initialized = False

    def analyseUserInput(x,y):
        """This function will be called in two cases:
            *by the next functoin
            *when there is some userInput stored from previous run

            This piece of code was refactored in order to be used in these two cases
            """
        global control_mouse
        #compute center of current contours and their distances to the user click
        #'cnts' here will be set on the loop that is written after this function definition
        centers = [jasf_cv.getCenterOfContour(c) for c in cnts]
        distances = [np.linalg.norm(np.array(c) - np.array((x,y))) for c in centers]
        #the mouse is the one closest to the user click
        i = np.argmin(distances)
        rx,ry = centers[i]
        mouse = cnts[i] 

        #the user cannot miss badly
        if jasf.math.pointDistance((rx,ry), (x,y)) > 20:
            print 'not close enough!'
            pass
        else: 
            print 'position set!'
            control_mouse.setPosition(rx, ry, mouse)
            control_mouse.initialized = True

            #add user input to dictionary of user inputs
            userInputData[control_settings['currentVideoFileName']].append({'frame':
                readControlSetting('framesSinceStart'), 'input':(rx,ry), 'settings_state':readSettingsState()}) 


    def onUserInputDblCklick(event, x, y, flags, params):
        """ mouse callback to set the rat position. This function gets the user press
        position and compare it with the known centers, picking the closest match. It will reject the chosen position if
        it is distant from the guessed centers"""
        global control_mouse
        if event == cv2.EVENT_LBUTTONDBLCLK:
            analyseUserInput(x,y)


    if modeReadingFromData:
        rx,ry = userData['input']
        setSettingsState(userData['settings_state'])

        ##sorry for this code repetition starting here
        #read parameters from settings window
        th, th_max, delta, dilateSize, erodeSize, LRA = readSettings()
        #select which image to use
        input = inputList[LRA]
        #get blue component
        input = input[:,:,1]
        input = input.copy()
        #find contours
        contourFinder.setParams(dilateSize, erodeSize, th, th_max)
        cnts, otsu_threshold, filterSmall = contourFinder.detectInterestingContours(input)
        ##code repetition ends here

        #call analyse; hopefully this will already set control_mouse.initialized and the loop will not run
        analyseUserInput(rx,ry)

    else:
        #in this case, the loop should run
        cv2.setMouseCallback('user input', onUserInputDblCklick)
    
    #ask user to select contour
    while  control_mouse.initialized == False:
        #read parameters from settings window
        th, th_max, delta, dilateSize, erodeSize, LRA = readSettings()
        #select which image to use
        input = inputList[LRA]
        #get blue component
        input = input[:,:,1]
        input = input.copy()
        #find contours
        contourFinder.setParams(dilateSize, erodeSize, th, th_max)
        cnts, otsu_threshold, filterSmall = contourFinder.detectInterestingContours(input)
        #draw all contours
        img2show = jasf_cv.drawContours(input, cnts)
        cv2.imshow('user input', img2show)
        ch = cv2.waitKey(150) & 0xFF

    cv2.destroyWindow('user input')


    #compute area and set the new thresholds
    updateValuesOfTh(delta)
    
def plotData(ax, x,y, legend="legenda", options=dict()):
    #clear the current axis
    ax.cla()
    #plot
    ax.plot(x, y, marker='o', linestyle='--', **options)
    ax.legend([legend])
    #ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')
#########################################
#Useful objects
#########################################

ret,frame = cam.read()
contourFinder = DevikasFilterGroundSubtraction_ContourFinder()
contourPicker = PreviousCenter_MousePicker()

#decide which flow computer to use
flowComputerAvailableOptions = ('regular', 'rangePercent_regular', 'rangePercent_windowFlow', 'windowFlow')
flowComputerOption = 'regular' if 'flowComputer' not in control_inputDict.keys() else control_inputDict['flowComputer']
flowComputer =                (   FlowComputer() if flowComputerOption == 'regular' else 
                                 rangePercentFlowComputer.RangePerCentFlowComputer_regularComputer() if flowComputerOption == 'rangePercent_regular' else  
                                 rangePercentFlowComputer.RangePerCentFlowComputer_windowFlow() if flowComputerOption == 'rangePercent_windowFlow' else 
                                 windowFlowUtil.WindowFlowComputer() if flowComputerOption == 'windowFlow' else FlowComputer() )
                            
print 'using the following class for flow computation', flowComputer.__class__

flowFilter_magnitude = flowUtil.FlowFilter()
flowFilter_conectivity = flowUtil.FlowFilter_ConnectedRegions()

#########################################
#Initialize plot for data init plot data 
#########################################
import matplotlib.pyplot as plt
plot, axis = plt.subplots(2,3)
xData = [[list() for j in range(axis.shape[1])] for i in range(axis.shape[0])]
yData = [[list() for j in range(axis.shape[1])] for i in range(axis.shape[0])]
def resetPlotData():
    for i in range(len(xData)):
        for j in range(len(xData[0])):
            xData[i][j] = list()
            yData[i][j] = list()
#used to keep track of iterations
iteration = 0
#initializate video and ask user for input
onVideoChange(0)
#########################################
#main loop
#########################################
while cam.isOpened():
    #-----------------------------------------------------------------
    #Step 0: decide what to do with input
    #-----------------------------------------------------------------
    waitTime = 5 if control_settings['control_mode'] == 'run' else 50
    ch = cv2.waitKey(waitTime) & 0xFF
    if ch == ord('q'):
        print 'finishing due to user input'
        break
    if ch == ord('p'):
        switchRunPause()
    if ch == ord('i'):
        askUserForInput(frame)
    if ch == ord('s'):
        control_show_plot = not control_show_plot
    if readControlSetting('control_mode') == 'run':
        ret,frame = cam.read()
        control_settings['framesSinceStart'] += 1 
        if control_settings['framesSinceStart'] in control_settings['listOfKnownFrames_with_input']:
            #read current video file name
            currentVideoFileName_aux = control_settings['currentVideoFileName']
            #find in the user input the one corresponding to the current frame; the returned type is a list
            userInputForThisFrame = filter(lambda x: x['frame'] == control_settings['framesSinceStart'], userInputData[currentVideoFileName_aux])
            askUserForInput(frame, True, userInputForThisFrame[0])
            #print a silly msg so we know something was done
            print 'aha! we avoided that!'
    if readControlSetting('control_mode') == 'pause':
        continue
    if ret == False:
        setControlSetting('control_mode', 'pause')
        continue

    #-----------------------------------------------------------------
    #Step 1: prepare input for algorith
    #-----------------------------------------------------------------
    #this is, pick the right half component and embed it into higher black image
    #-----------------------------------------------------------------
    th, th_max, delta, dilateSize, erodeSize, LRA = readSettings()
    #select which image to use
    left, right = devika_cv.break_left_right(frame)
    inputList = [left, right, frame]
    input = inputList[LRA]
    #embed input image into a bigger black image
    h,w,d = input.shape
    ho, wo = 40,40 # 'o' for ofset; those are the dimension of the margin
    extendedInput = np.zeros((h+2*ho,w+2*wo,d), dtype=np.uint8)
    extendedInput[ho:ho+h, wo:wo+w, :] = input[:,:,:]

    #get blue component
    B = input[:,:,0]
    Bextended = extendedInput[:,:,0]

    #-----------------------------------------------------------------
    #Step 2: find candidates to rat contour
    #-----------------------------------------------------------------
    input = B.copy()
    contourFinder.setParams(dilateSize, erodeSize, th, th_max)
    contours, otsu_threshold, filterSmall = contourFinder.detectInterestingContours(input)

    #-----------------------------------------------------------------
    #Step 3: select which contour is the real mouse
    #-----------------------------------------------------------------
    rx,ry,new_mouse = contourPicker.pickCorrectContour(contours, {'last_center':(control_mouse.rx, control_mouse.ry), 'distanceRejectTh':2000})
    #if mouse was found, update parameters; testing the type here is checking for errors
    if type(new_mouse) is not bool:
        control_mouse.setPosition(rx, ry, new_mouse)
        updateValuesOfTh(delta)

    #convert mouse coordinates to extended frame
    offset = np.empty_like(control_mouse.mouse)
    offset.fill(40)#the ofset was 40,40 up there [sorry for the magic number]
    translatedMouse = control_mouse.mouse + offset

    #-----------------------------------------------------------------
    #Step 3.1: find angle orientation of the mouse
    #this is done by looking at the rotation angle of the minimum bouding box arround the
    #countourn of the mouse
    #-----------------------------------------------------------------
    #get angle of rotation in degrees
    minimumAreaRectAroundMouse_center, _wh, minimumAreaRectAroundMouse_angle = cv2.minAreaRect(translatedMouse)
    minimumAreaRectAroundMouse_center = np.array(minimumAreaRectAroundMouse_center).astype(np.int32).reshape(1,2)
    #-----------------------------------------------------------------
    #Step 4: some drawing of the selected Mouse
    #also draw an arrow poiting at the minimum area rectangle orientation
    #-----------------------------------------------------------------
    #draw countours
    output = Bextended.copy()
    #draw 60x60 rectangle around mouse and other countourn boxes
    output = jasf_cv.drawContours(output, [translatedMouse], fixedDimRect=True, fdr_dim=np.array((60,60)), fdr_color = (200,0,200))
    output = jasf.cv.drawFixedLenghtArrow(output, minimumAreaRectAroundMouse_center, np.deg2rad(minimumAreaRectAroundMouse_angle), 20)
    #get fixed lenght (60,60) rectangle image of mouse
    mouseImg = jasf.cv.getRoiAroundContour(extendedInput, translatedMouse, dim = np.array((60,60)))
    #-----------------------------------------------------------------
    #Step 5: compute optical flow
    #-----------------------------------------------------------------
    #The flow is computed on a dense grid that fills a square around the mouse
    #The flow is computed in relationship to the global extended frame 
    #-----------------------------------------------------------------
    #compute grid to look for flow
    x,y,w,h = cv2.boundingRect(translatedMouse)
    X,Y = np.mgrid[x:x+w, y:y+h]
    grid = np.array(np.vstack((X.flatten(),Y.flatten())).transpose(), dtype=np.float32) 
    flowComputer.setGrid(grid)

    if 'rangePercent' in flowComputerOption:
        minPercent,maxPercent = readRangePercentSettings()
        flowComputer.setPercentageInterval(minPercent/100.0,maxPercent/100.0)


    #find flow; the output wil be the start and end point of every flow vector
    flowInput = Bextended.copy()
    oldP, newP = flowComputer.apply(flowInput)
    if oldP.size == 0:
        continue
    #-----------------------------------------------------------------
    #Step 5.1: Filter Flow or flow filter
    #-----------------------------------------------------------------
    flowLowTh, flowUpTh,flowCLowTh, flowCUpTh, connectFilterOn, magFilterOn  = readFlowSettings()
    #filter my magnitude
    if magFilterOn:
        flowFilter_magnitude.setTh(flowLowTh, flowUpTh)
        oldP, newP = flowFilter_magnitude.apply(oldP, newP)
    #filter by connectivity
    if connectFilterOn:
        flowFilter_conectivity.setTh(flowCLowTh, flowCUpTh)
        oldP, newP = flowFilter_conectivity.apply(oldP, newP)
    #-----------------------------------------------------------------
    #Step 6.0 Processing Flow or process flow 
    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    #Step 6.1 find average flow
    #-----------------------------------------------------------------
    averageOld, averageNew = rangePercentFlowComputer.averageFlow(oldP, newP)
    averageFlow_norm = jasf.math.pointDistance(averageOld, averageNew)
    averageFlow = averageNew - averageOld
    averageFlow_angle = np.rad2deg(np.arctan2(averageFlow[0,1], averageFlow[0,0]))

    #-----------------------------------------------------------------
    #Step 7.0: Plotting plot data 
    #-----------------------------------------------------------------
    #plot magnitude and angle of mean flow
    iteration += 1
    if iteration % 10 == 0 and control_show_plot:
        #we only plot every 10 iterations so we don't slow down the program too much. Also, we reduce by 10 the number
        #of points being ploted.
        currentFrameNumber = control_settings['framesSinceStart']

        #plot magnitude of mean flow
        xData[0][0].append(currentFrameNumber)
        yData[0][0].append(averageFlow_norm)
        plotData(axis[0,0], xData[0][0], yData[0][0], 'magnitude', {'color':'blue'})

        #plot angle of mean flow
        xData[0][1].append(currentFrameNumber)
        yData[0][1].append(averageFlow_angle)
        plotData(axis[0,1], xData[0][1], yData[0][1], 'angle', {'color':'green'})

        #plot angle of minimum bounding rectangle
        xData[1][0].append(currentFrameNumber)
        yData[1][0].append(minimumAreaRectAroundMouse_angle)
        plotData(axis[1,0], xData[1][0], yData[1][0], 'rotation angle', {'color':'red'})

        #pause to allow matplot lib to show data
        plt.pause(0.000005)

    if len(yData[0][0]) == 100:
        #we only show 100 points, so once the vector is complete we discard the first component
        pass
        #plot_y = plot_y[1:]
        #plot_x = plot_x[1:]


    #plotting fft
    #the fft is computed with data from every iteration, while the plot shows only data every 10 iterations
    data2FFT.append(averageFlow_norm)
    if control_show_fft_fft and len(data2FFT) == control_array['frames2FFT']:
        #this is the size of the computed fft
        fft_N = 100
        #we need to shift to make sure the zero frequency is centered
        f = np.fft.fftshift(jasf.math.fft(data2FFT, fft_N))
        freq = np.fft.fftshift(np.fft.fftfreq(fft_N))

        plotData(axis[0,1], freq, f)
        #we pause so the screen has time to update(apparently this is neccessary)
        plt.pause(0.000005)

        #discard the data that was just processed
        data2FFT = []

    #-----------------------------------------------------------------
    #Step 8.0 Draw flow 
    #-----------------------------------------------------------------
    #draw every valid flow vector with magnitude > 0.2 as BLUE
    #the flow of maximum magnitude is drawn GREEN and the mean flow is RED
    #-----------------------------------------------------------------
    mouseImg = flowUtil.draw_flow(flowInput, oldP, newP, jasf.cv.blue, 1, p=1, q=2, th=0.2, drawArrows=True)
    #mouseImg = flowUtil.draw_flow(mouseImg, averageOld.reshape(1,1,2), averageNew.reshape(1,1,2), jasf.cv.red, 2, drawArrows=True)

    mouseImg = jasf.cv.drawFixedLenghtArrow(mouseImg, averageOld.reshape(1,1,2), np.deg2rad(averageFlow_angle), 30, color=jasf.cv.red)
    #-----------------------------------------------------------------
    #Step 8.0: Show images 
    #-----------------------------------------------------------------
    jasf.cv.imshow(['B', 'otsuTh', 'clean+filter', 'tracking', 'flowImg'],\
            [B, 255*otsu_threshold, 255*filterSmall, output, mouseImg])



#store state of system
paramDict = readSettingsState()
print 'here is the current configuration', paramDict
with open('./paramDict.json', 'w') as f:
    print 'saving state of trackbars to', f.name
    json.dump(paramDict, f)

with open('./userInputData.json', 'w') as f:
    print 'saving state of user input to', f.name
    json.dump(userInputData, f)
cv2.destroyAllWindows()
cam.release()
