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
from inputParse import parseInput

#videos will be read from this path
control_Path2VideoFiles = '../videoCleanExperiment/output/fallingMouse/'
control_Path2VideoFiles = '../../video/mp4/'
control_settings = {'control_mode':'run'}
control_inputDict = parseInput(sys.argv)

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
    setControlSetting('control_mode', 'run')
    askUserForInput(frame)

jasf_cv.setTrackbar('video file', 0, len(videoFiles)-1, onCallBack = onVideoChange, window_name='settings')
jasf.cv.setManyTrackbars(['th', 'max', 'delta', 'dilateSize', 'erodeSize', 'LRA'], [0, 100, 4, 5, 0, 1], [400, 400, 10, 21, 21, 2])
jasf.cv.setManyTrackbars(['flow_lowTh', 'flow_upTh', 'flowConect_lowTh', 'flowConect_upTh'], [2, 30, 160, 10000], [50, 50, 1000, 10000])
jasf.cv.setManyTrackbars(['connectivityFilterOn', 'magnitudeFilterOn', 'class1', 'class2', 'class3'], [0, 0, 95, 85, 75], [1,1,100,100,100], i=1)

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
#####################################
#Auxiliar Functions
#####################################
def setControlSetting(name, val):
    global control_settings
    control_settings[name] = val

def readControlSetting(name):
    global control_settings
    return control_settings[name]

def readSettings():
    """ read general settings """
    return jasf.cv.readManyTrackbars(['th', 'max', 'delta', 'dilateSize', 'erodeSize', 'LRA'])

def readClassSettings():
    return jasf.cv.readManyTrackbars(['class1', 'class2', 'class3'], i=1)

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

def askUserForInput(frame):
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
    def onUserInputDblCklick(event, x, y, flags, params):
        """ mouse callback to set the rat position. This function gets the user press
        position and compare it with the known centers, picking the closest match. It will reject the chosen position if
        it is distant from the guessed centers"""
        global control_mouse
        if event == cv2.EVENT_LBUTTONDBLCLK:
            #compute center of current contours and their distances to the user click
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
    
def plotData(ax, x,y):
    #clear the current axis
    ax.cla()
    #plot
    ax.plot(x, y, marker='o', linestyle='--')
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

    #-----------------------------------------------------------------
    #Step 4: some drawing of the selected Mouse
    #-----------------------------------------------------------------
    #draw countours
    output = Bextended.copy()
    #convert mouse coordinates to extended frame
    offset = np.empty_like(control_mouse.mouse)
    offset.fill(40)#the ofset was 40,40 up there [sorry for the magic number]
    translatedMouse = control_mouse.mouse + offset
    #draw 60x60 rectangle around mouse 
    output = jasf_cv.drawFixedDimAroundContourCenter(output, [translatedMouse], (200, 0, 200), np.array((60,60)))
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

    #find flow; the output wil be the start and end point of every flow vector
    flowInput = Bextended.copy()
    oldP, newP = flowComputer.apply(flowInput)
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
    #Step 6.1 keep track of the highest value flow
    #-----------------------------------------------------------------
    #compute flow magnitudes
    flow = newP - oldP
    flowNorm = np.linalg.norm(flow, axis = 1)
    #the following condition is necessary because not always the flow computation is successul.
    #At the 1st iteration for example, there is no flow to compute
    maxFlowNorm = flowNorm.max() if flow.shape[0] > 0 else 0.0
    #will need the max flow index to paint it differently later
    maxFlow_i = flowNorm.argmax() if maxFlowNorm > 0.0 else -1

    maxFlowOld, maxFlowNew = (oldP[maxFlow_i], newP[maxFlow_i]) if maxFlow_i > 0 else (np.zeros((2,1)), np.zeros((2,1)))

    #-----------------------------------------------------------------
    #Step 6.1 find average flow
    #-----------------------------------------------------------------
    averageOld, averageNew = rangePercentFlowComputer.averageFlow(oldP, newP)

    #-----------------------------------------------------------------
    #Step 7.0: Plotting
    #-----------------------------------------------------------------
    #plot highest magnitude
    iteration += 1
    if iteration % 10 == 0 and control_show_plot:
        #we only plot every 10 iterations so we don't slow down the program too much. Also, we reduce by 10 the number
        #of points being ploted.
        xData[0][0].append(iteration)
        yData[0][0].append(maxFlowNorm)
        plotData(axis[0,0], xData[0][0], yData[0][0])
        plt.pause(0.000005)
    if len(yData[0][0]) == 100:
        #we only show 100 points, so once the vector is complete we discard the first component
        pass
        #plot_y = plot_y[1:]
        #plot_x = plot_x[1:]


    #plotting fft
    #the fft is computed with data from every iteration, while the plot shows only data every 10 iterations
    data2FFT.append(maxFlowNorm)
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
    mouseImg = flowUtil.draw_flow(mouseImg, maxFlowOld.reshape(1,1,2), maxFlowNew.reshape(1,1,2), jasf.cv.green, 2, drawArrows=True)
    mouseImg = flowUtil.draw_flow(mouseImg, averageOld.reshape(1,1,2), averageNew.reshape(1,1,2), jasf.cv.red, 2, drawArrows=True)
    #-----------------------------------------------------------------
    #Step 8.0: Show images 
    #-----------------------------------------------------------------
    jasf.cv.imshow(['B', 'otsuTh', 'clean+filter', 'tracking', 'flowImg'],\
            [B, 255*otsu_threshold, 255*filterSmall, output, mouseImg])


cv2.destroyAllWindows()
cam.release()
