import sys
sys.path.insert(0, '../')
sys.path.append('../lineWiseMacineLearningExperiment/')
import numpy as np
import cv2
from cv2 import imshow
import jasf
from jasf import jasf_cv
from jasf import jasf_ratFinder
from devika import devika_cv
import pathlib

from multiFiltersDataStructures import getGBR_componentOperation

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
jasf_cv.getNewWindow('input')
jasf_cv.getNewWindow('output')
jasf_cv.getNewWindow('settings')
#####################################
#set trackbars
#####################################
w,h = jasf.cv.getVideoCaptureFrameHeightWidth(cam)

jasf_cv.setTrackbar('video file', 0, len(videoFiles)-1, onCallBack = onVideoChange, window_name='settings')
jasf.cv.setManyTrackbars(['row', 'column', 's'], [133, 100,5], [h-1, w-1,20])
jasf_cv.setTrackbar('th', 32, 120)

def readSettings():
    return jasf.cv.readManyTrackbars(['row', 'column', 's', 'th'])

bGetter = getGBR_componentOperation('b')

#########################################
#Initialize plot for data
#########################################
import matplotlib.pyplot as plt
plot, ax = plt.subplots(2,4)
ax[0][0].set_xlabel('index in row')
ax[1][0].set_xlabel('index in column')

for axis in ax:
    for axx in axis:
        axx.autoscale(enable=True, axis = 'both')

def operationOnLineDate(data, s):
    """
    compute standard deviation of window i+-s around pixel i; for every pixel from s to i+s
    """
    output = [np.std(data[i-s:i+s+1]) for i in range(s, len(data)-s)]
    return output


def plotData(ax, y):
    ax.cla()
    ax.plot(range(len(y)), y, marker='o', linestyle='--')
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')


def updatePlot(input, r,c,s,th):
    """
    (r,c) for row and column, s for size of window and th for threshold
    """
    #generate row data and processing
    rowData = input[r,:].reshape(-1).tolist()
    std_rowData = operationOnLineDate(rowData, s)
    th, th_rowData = cv2.threshold(np.array(std_rowData, dtype=np.uint8), th, 1, cv2.THRESH_BINARY_INV)

    #find points of jumps from low to high
    padded = np.vstack((th_rowData,np.array(0, dtype=np.int32)))
    shifted = np.roll(padded, 1)
    diff = padded - shifted
    diff = diff.reshape(-1)
    x = np.arange(len(diff), dtype = np.int32)
    x_low2High = x[diff == 1]

    #find points of jump from high to low
    padded = np.vstack((np.array(0, dtype=np.int32),th_rowData))
    shifted = np.roll(padded, -1)
    diff = padded - shifted
    diff = diff.reshape(-1)
    x = np.arange(len(diff), dtype = np.uint32) - 1
    x = x[diff == 1]
    x_high2low = x

    #plot row data
    plotData(ax[0,0], rowData)
    plotData(ax[0,1], std_rowData)
    ax[0,1].axhline(y = th, color='k', linewidth=2)
    plotData(ax[0,2], th_rowData)
    for xx in x_low2High:
        ax[0,2].axvline(x=xx, color = 'r', linewidth=1)
    for xx in x_high2low:
        ax[0,2].axvline(x=xx, color = 'g', linewidth=1)


    #generate col data
    colData = input[:,c].reshape(-1).tolist()
    std_colData = operationOnLineDate(colData, s)
    th, th_colData = cv2.threshold(np.array(std_colData, dtype=np.uint8), th, 1, cv2.THRESH_BINARY_INV)

    #plot col data
    plotData(ax[1,0], colData)
    plotData(ax[1,1], std_colData)
    ax[1,1].axhline(y = th, color='k', linewidth=2)
    plotData(ax[1,2], th_colData)


    plt.pause(0.0005)


def detect(input, s, th):
   pass 

counter = 0
r,c,s,th = readSettings()
while cam.isOpened():
    counter += 1
    waitTime = 5 if control_mode == 'run' else 50
    ch = cv2.waitKey(waitTime) & 0xFF
    if ch == ord('q'):
        print 'finishing due to user input'
        break
    if ch == ord('p'):
        control_mode = 'run' if control_mode == 'pause' else 'pause'
    if control_mode == 'run':
        ret,frame = cam.read()
    if ret == False:
        control_mode = 'pause'
        continue

    
    old_r, oc, os, oth = r,c,s,th
    r,c,s,th = readSettings()
    input = bGetter.apply(frame)

    if np.any(np.array((old_r,oc,os,oth)) != np.array((r,c,s,th))) or counter == 1:
        updatePlot(input, r,c,s,th)


    input = jasf_cv.convertGray2BGR(input)
    input[r,:,:]  = 0 
    input[:,c,:]  = 0
    input[r,:,:] += np.array((255,0,0))
    input[:,c,:] += np.array((0,0,255))

    cv2.imshow('input', input)

cv2.destroyAllWindows()
cam.release()
