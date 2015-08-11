import sys
sys.path.insert(0, '../')
import cv2
import numpy as np
from matplotlib import pyplot as plt
from jasf import jasf_cv

def myLineFilter(lines, max):
    if lines == None:
        return lines
    myLines = []
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if np.abs(x2 - x1) < max:
            myLines.append(line)

    return myLines

cam = cv2.VideoCapture('../../video/mp4/myFavoriteVideo.mp4')
#cam = cv2.VideoCapture(0)
outputWindow = jasf_cv.getNewWindow('Probabilistic Hough Line')
track_window = jasf_cv.getNewWindow('settings')
cannyWindow = jasf_cv.getNewWindow('Canny')
houghSettings_window = jasf_cv.getNewWindow('HoughLineSettings')



def onControlChange(opt):
    if opt == 0:
        print 'changing mode to Probabilistic Hough Transform'
    else:
        print 'changing mode to Standard Hough Transform'
    control_hough_mode = opt

#general settings 
jasf_cv.setTrackbar('blur_size', 3, 17)
jasf_cv.setTrackbar('th_min', 100, 255)
jasf_cv.setTrackbar('th_max', 150, 255)
jasf_cv.setTrackbar('Prob/Standard Hough', 1, 1, onCallBack = onControlChange)
jasf_cv.setTrackbar('th', 80, 255)
jasf_cv.setTrackbar('minLength', 50, 200)
jasf_cv.setTrackbar('maxLineGap', 5, 20)


#settings for hough transform
jasf_cv.setTrackbar('minAngle', 0, 30, window_name = houghSettings_window)
jasf_cv.setTrackbar('maxAngle', 5, 30, window_name = houghSettings_window)
jasf_cv.setTrackbar('myFilter?', 1, 1, window_name = houghSettings_window)
jasf_cv.setTrackbar('myFilter_max', 54, 100, window_name = houghSettings_window)

def readSettings():
    th = cv2.getTrackbarPos('th', track_window)
    minLength= cv2.getTrackbarPos('minLength', track_window)
    maxGap = cv2.getTrackbarPos('maxLineGap', track_window)
    th_min = cv2.getTrackbarPos('th_min', track_window)
    th_max = cv2.getTrackbarPos('th_max', track_window)
    blur_size = cv2.getTrackbarPos('blur_size', track_window)
    control_hough_mode = cv2.getTrackbarPos('Prob/Standard Hough', track_window)
    minAngle = cv2.getTrackbarPos('minAngle', houghSettings_window)
    maxAngle = cv2.getTrackbarPos('maxAngle', houghSettings_window)
    control_myFilter = cv2.getTrackbarPos('myFilter?', houghSettings_window)
    myFilter_max= cv2.getTrackbarPos('myFilter_max', houghSettings_window)

    return th, minLength, maxGap, th_min, th_max, blur_size, control_hough_mode, minAngle, maxAngle, control_myFilter, myFilter_max

cannyConfiguration = {'apertureSize':3, 'L2gradient':True}
houghPConfiguration = {'rho':1, 'theta':np.pi/180, 'threshold':-1,\
        'minLineLength':-1, 'maxLineGap':-1}
houghConfiguration = {'rho':1, 'theta':np.pi/180, 'threshold':-1,\
        'min_theta':-1, 'max_theta':-1}

while cam.isOpened():
    ret, frame = cam.read()
    if ret == False:
        print 'Ending execution due to bad reading of video!'
        break
    B,G,R = cv2.split(frame)


    th, minLength, maxGap, th_min, th_max, blur_size, control_hough_mode, minAngle,\
            maxAngle, control_myFilter, myFilter_max = readSettings()
    houghConfiguration


    blurKernel = (blur_size, blur_size)
    input = G if blur_size == 0 else cv2.blur(G, blurKernel)


    canny = cv2.Canny(input,th_min,th_max, **cannyConfiguration) 

    if control_hough_mode == 0 :
        houghPConfiguration['threshold'] = th
        houghPConfiguration['minLineLength'] = minLength
        houghPConfiguration['maxLineGap'] = maxGap
        lines = cv2.HoughLinesP(canny, **houghPConfiguration)
        lineDetection = jasf_cv.drawLines_endPointsInput(input, lines)

    if control_hough_mode == 1:
        houghConfiguration['threshold'] = th
        houghConfiguration['min_theta'] = minAngle
        houghConfiguration['max_theta'] = maxAngle
        lines = cv2.HoughLines(canny, **houghConfiguration)
        if control_myFilter == 1:
            lines = myLineFilter(lines, myFilter_max)
        lineDetection = jasf_cv.drawLines_LineDataStructure(input, lines)


    cv2.imshow(outputWindow,lineDetection)
    cv2.imshow(cannyWindow,canny)

    ch = cv2.waitKey(5) & 0xFF
    if ch == ord('q'):
        print "execution being terminated due to press of key 'q'"
        break

cam.release()
cv2.destroyAllWindows()
