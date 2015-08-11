import numpy as np
import cv2
from scipy import ndimage

def doNothing(val):
    pass
class math:
    def __init__(self):
        pass

    @staticmethod
    def pointDistance(a,b):
        """Compute euclidian distance of two vectors"""
        return np.linalg.norm(np.array(a)-np.array(b))

    @staticmethod
    def fft(y, n):
        """ return the absotule value of the FFT of a real sequence. This is what we need for this project.
        Returned array is shaped to a column vector (n,1)"""
        return np.abs(np.fft.fft(y, n=n)).reshape(-1,1) 
    

class jasf_ratFinder:
    def __init__(self):
        pass

    detectFloor_cleaning_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    @staticmethod
    def detectFloor(input, alpha, previous_approx=[], previous_roi=[], allowed_jump=0):
        """Run a set of filters based on Devika's code, invert it, find contours and
        return the one with the heighest area. Uses countours of previous detection to
        select weather or not the current detection is acceptable. If it is not,return the
        orld countour"""
        #leave Otsu decide the threshold
        otsu_th, otsu_threshold = cv2.threshold(input, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #opening operation to fill holes and eliminate noise
        open = cv2.morphologyEx(otsu_threshold, cv2.MORPH_OPEN, jasf_ratFinder.detectFloor_cleaning_kernel)
        open = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

        #invert image
        ret, invert = cv2.threshold(open, 0.5, 1, cv2.THRESH_BINARY_INV)
        if len(previous_roi) > 0:
            invert = cv2.drawContours(invert, [previous_roi], 0, 1, -1)
        #find countours
        ret, cnts, ret2 = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #find approximations to degree determined by epsilon
        approx = [cv2.approxPolyDP(c, alpha*cv2.arcLength(c,True), True) for c in cnts] 
        #the floor is the one with the highest value of area
        roi = approx[np.argmax([cv2.contourArea(c) for c in approx])]

        #analyse if the new detection is acceptable
        #if it is not, return the previous detection as the new one.
        prev_size = len(previous_roi)
        if (len(previous_roi) > 0) and ((len(roi) not in range(prev_size-2, prev_size+2+1)) or\
                (np.abs(cv2.contourArea(roi) - cv2.contourArea(previous_roi)) > allowed_jump)):
            return previous_roi, previous_approx, invert, otsu_th

        return roi, approx, invert, otsu_th

    #the following method is kept here for reasons of providing compatibility to older code
    #That is, this code is not up to date
    @staticmethod
    def detectInterestingContours(input, th, th_max, otsu_th):
        """Apply a set of filtering techniques suggested by Devika and return contours and
        some of the images produced in the process so one can visualize outputs of the
        filters. Small objects are removed. Only objects with area between th and th_max
        are candidates to be returned. Chech function areaBandPassObjectFilter to learn
        more."""
        global cleaning_kernel
        #leave Otsu decide the threshold
        ret, otsu_threshold = cv2.threshold(input, otsu_th, 1, cv2.THRESH_BINARY)
        #opening operation to fill holes and eliminate noise
        open = cv2.morphologyEx(otsu_threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
        open = cv2.dilate(open, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        #filter small objects
        filterSmall = jasf_cv.areaBandPassObjectFilter(open.copy(), th, th_max)
        target = filterSmall.copy()
        output, contours, hier = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, otsu_threshold, open, filterSmall

class cv:
    red = (0,0,255)
    blue = (0,255,0)
    green = (255,0,0)

    def __init__():
        raise Exception('This class should no be instantiated!')

    @staticmethod
    def getVideoCaptureFrameHeightWidth(cap):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w,h

    @staticmethod
    def waitKey(time):
        return cv2.waitKey(time) & 0xFFFF
    @staticmethod
    def readTrackbar(varName, window_name = 'settings', i=0):
        if i>0:
            window_name = window_name + str(i)
        return cv2.getTrackbarPos(varName, window_name)

    @staticmethod
    def readManyTrackbars(nameList, windowName = 'settings', i = 0):
       return [cv.readTrackbar(name, windowName, i=i) for name in nameList]

    @staticmethod
    def imshow(nameList, imgList):
        for j,name in enumerate(nameList):
            cv.imshow(name, imgList[j])
            
    @staticmethod
    def setManyTrackbars(nameList, startList, maxList, windowName = 'settings',i=0):
        for j,name in enumerate(nameList):
            jasf_cv.setTrackbar(name, window_name=windowName, start = startList[j], max = maxList[j], i=i)

    @staticmethod
    def setManyTrackbarsPos(nameList, posList, windowName = 'settings', i=0):
        for j,name in enumerate(nameList):
            cv.setTrackbarPos(name, posList[j], windowName,i=i)

    @staticmethod
    def setTrackbarPos(varName, pos, window_name = 'settings',i=0):
        if i>0:
            window_name = window_name + str(i)
        cv2.setTrackbarPos(varName, window_name, pos)

    @staticmethod
    def switchBinnaryTrackbar(varName, windowName = 'settings',i=0):
        currentVal = cv.readTrackbar(varName,i=i)
        cv.setTrackbarPos(varName, 0 if currentVal == 1 else 1, i=i)

    @staticmethod
    def equalizeHist(imgList):
        out = [cv2.equalizeHist(img) for img in imgList]
        return out

    @staticmethod
    def inRange(imgList, lowerb, upperb):
        out = [cv2.inRange(img, lowerb, upperb) for img in imgList]
        return out

    @staticmethod
    def getManyWindows(win_name_array, dim=(350,300), n=(3,3)):
        return [jasf_cv.getNewWindow(name, dim, n) for name in win_name_array]

    @staticmethod
    def imshow(winame_list, img_list):
        for i,_ in enumerate(winame_list):
            cv2.imshow(winame_list[i], img_list[i])

    @staticmethod
    def binarize(img, th = 100):
        return cv2.threshold(img, th, 1, cv2.THRESH_BINARY)

    @staticmethod
    def invertBoolean(img):
        """ zeros become ones, ones become 0"""
        return cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY_INV) 

    @staticmethod
    def getRoiAroundContour(img, cnt, dim = np.array((30,30))):
        x,y,w,h = cv2.boundingRect(cnt)
        center = jasf_cv.getCenterOfContour(cnt) 
        lt = center - 0.5*dim
        rb = center + 0.5*dim
        x0,y0 = (int(lt[0]), int(lt[1]))
        x1,y1 = (int(rb[0]), int(rb[1]))
        roi = img[y0:y1, x0:x1]
        return roi.copy()

class jasf_cv:
    def __init__(self):
        pass

    @staticmethod
    def drawContours(output, cnts, axisParalelRect=True, apr_color = (0,255,0),\
            minAreaRect=True, mar_color = (0,0,255),\
            centerCircle=True, cc_color=(0,0,255),\
            fixedDimRect=False, fdr_color = (255,0,0), fdr_dim = np.array((30,30))):
        """Draw contours and both axis parales and not axis pararel bouding rectangle. Also draw a circle in the center
        of the contour"""
        #makes sure we're dealing with 3 chanel image
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        #draw raw contours
        cv2.drawContours(output, cnts, -1, (255,0,0), 2)
        for c in cnts:
            if len(c) == 0:
                continue
            #draw axis pararel bounding rectangle
            if axisParalelRect:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(output, (x,y), (x+w, y+h), apr_color, 2)
            #draw min are bouding rectangle
            if minAreaRect:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output, [box], 0, mar_color, 2)
            #draw center of contour
            if centerCircle:
                center = jasf_cv.getCenterOfContour(c) 
                cv2.circle(output, center, 2, cc_color, 2)
            if fixedDimRect:
                center = jasf_cv.getCenterOfContour(c) 
                lt = center - 0.5*fdr_dim
                rb = center + 0.5*fdr_dim
                lt = (int(lt[0]), int(lt[1]))
                rb = (int(rb[0]), int(rb[1]))
                cv2.rectangle(output, lt, rb, fdr_color, 2)

        return output


    @staticmethod
    def drawFixedDimAroundContourCenter(img, cnts, color, dim = np.array((30,30))):
        return jasf_cv.drawContours(img, cnts, axisParalelRect=False, minAreaRect=False,centerCircle=False,
                fixedDimRect=True, fdr_color = color, fdr_dim=dim)


    @staticmethod
    def getStructuringRectangle(size):
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    @staticmethod
    def computeDenseOpticalFlow(img_prev, img_next):
        """Receive two gray images of same size and compute optical flow between them"""
        return cv2.calcOpticalFlowFarneback(\
            prev=img_prev, next=img_next, flow=None,\
            pyr_scale = 0.5, levels=3, winsize=15,\
            iterations=3, poly_n=5, poly_sigma=1.2,\
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    @staticmethod
    def convertBGR2Gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def convertGray2BGR(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def showImage(img, name='test'):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(name)

    @staticmethod
    def showVideo(filename, windowName='test', waitTime=5):
        cam = cv2.VideoCapture(filename)
        while cam.isOpened():
            ret, frame = cam.read()
            cv2.imshow(windowName, frame)
            ch =  cv2.waitKey(waitTime) & 0xFF
            if ch == ord('q'):
                break
        cam.release()
        cv2.destroyWindow(name)


    lastWindowIndex = -1
    @staticmethod
    def resetIndexes():
        jasf_cv.lastWindowIndex = -1

    control_mode_jasf_cv_silence = False

    @staticmethod
    def switchSilentMode():
        jasf_cv.control_mode_jasf_cv_silence = not jasf_cv.control_mode_jasf_cv_silence

    @staticmethod
    def getNewWindow(name=None, dimension=(350,300), n=(3,3)):
        jasf_cv.lastWindowIndex = jasf_cv.lastWindowIndex + 1
        i = jasf_cv.lastWindowIndex
        if name == None:
            windowName = str(i)
        else:
            windowName = name
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

        x_step = dimension[0]
        y_step = dimension[1]
        x_n = n[0]
        y_n = n[1]
        x = x_step*int(np.mod(i,x_n))
        y = y_step*int(np.floor(i/y_n))
        cv2.moveWindow(windowName, x,y)
        cv2.resizeWindow(windowName, x_step, y_step)

        if jasf_cv.control_mode_jasf_cv_silence == False:
            print 'window', windowName, 'moved to position', x,y
        return windowName

    @staticmethod
    def getSettingsWindow(n=(3,3)):
        return jasf_cv.getNewWindow('settings', n = n)

    @staticmethod
    def getInputWindow():
        return jasf_cv.getNewWindow('input')

    @staticmethod
    def getOutputWindow():
        return jasf_cv.getNewWindow('output')

    @staticmethod
    def getBasicWindows():
        """ set input, settings and output windows. Most programs have this basic setup"""
        return jasf_cv.getInputWindow(),jasf_cv.getSettingsWindow(),jasf_cv.getOutputWindow()

    @staticmethod
    def setTrackbar(varName, start=100, max=255, window_name = 'settings', onCallBack=doNothing,i=0):
        if i >0:
            window_name = window_name + str(i)
        cv2.createTrackbar(varName, window_name, start, max, onCallBack)
        return varName


    @staticmethod
    def drawLines_endPointsInput(img, lines, color=(0,0,255), thickness=2):
        if lines == None:
            return img
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img, (x1,y1), (x2,y2), color, thickness)  

        return img

    @staticmethod
    def drawLines_LineDataStructure(img, lines, color = (0,0,255), thickness=2):
        if lines == None:
            return img
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

            cv2.line(img,(x1,y1),(x2,y2),color,thickness)
            cv2.circle(img, (x0,y0), 3, (255,0,0))
        return img
    @staticmethod
    def fill(img):
        input = img.copy()
        im2, countour, hier = cv2.findContours(input, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for ctr in countour:
            cv2.drawContours(img, [ctr], 0, 255, -1)

        return img

    @staticmethod
    def areaBandPassObjectFilter(img, th=5, max=150):
        """ img received should be binnary. Draw a filled black contour around objects of
        small area. Only objects with area between th and max are kept"""
        target = img.copy()
        output, contours, hier = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blackList = []
        for c in contours:
            area = cv2.contourArea(c)/100
            if area >= th and area <= max:
                pass
            else:
                blackList.append(c)
        #paint all black listed objects with black
        cv2.drawContours(img, blackList, -1, 0, cv2.FILLED)
        return img.astype(np.uint8)

    @staticmethod
    def fill(img):
        """both img and element should be binnary"""
        e = np.ones((3,3), np.int)
        e[1,1] = 0
        interiorPoints = ndimage.binary_hit_or_miss(img, e).astype(np.int)
        img = img + interiorPoints
        return img.astype(np.uint8) 

    @staticmethod
    def clean(img):
        """both img and element should be binnary"""
        e = np.zeros((3,3), np.int)
        e[1,1] = 1
        interiorPoints = ndimage.binary_hit_or_miss(img, e).astype(np.int)
        img = img - interiorPoints
        return img.astype(np.uint8)

    @staticmethod
    def countContours(img):
        input = img.copy()
        output, contours, hier = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)

    @staticmethod
    def thicken(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #find initial number of objects
        n = jasf_cv.countContours(img)
        
        for i in range(20):
            #keep track of the previous img
            previous = img.copy()
            img = cv2.dilate(img, kernel, iterations = 1)
            if jasf_cv.countContours(img) != n:
                return previous
        return img
            
    @staticmethod
    def getCenterOfContour(c):
        M = cv2.moments(c)
        cx = np.int(M['m10']/M['m00'])
        cy = np.int(M['m01']/M['m00'])
        return (cx, cy)

    @staticmethod
    def isContourQuadrilaterum(contour):
        pass
