import sys
sys.path.insert(0, '../')
import cv2
import numpy as np

class FilterDealer(object):
    """This class holds a set of operations that will aplied all independently to the same input image"""
    def __init__(self):
        self.filters = []

    def addFilter(self, f):
        self.filters.append(f)

    def getNumberOfFilters(self):
        return len(self.filters)

    def apply(self, img, paramsDictList):
        return [f.apply(img, paramsDictList[i]) for i,f in enumerate(self.filters)]

class abstractFilterOperation(object):
    """This kind of operation is one that is usually applied with many others to the same input."""
    def __init__(self):
        pass
    def apply(self, img, paramsDict):
        """ Receives a GBR image and return a grayscale image of same size as input"""
        pass

class getGBR_componentOperation(abstractFilterOperation):
    def __init__(self, component='g'):
        componentDict = {'g':0, 'b':1, 'r':2}
        self.component = componentDict[component]

    def apply(self, img, paramsDict=dict()):
        return img[:,:,self.component]

class paint4Horizontals4VerticalsOperation(abstractFilterOperation):
    def __init__(self):
        pass

    def apply(self, img, paramsDict):
        #extract parameters from dict    
        horizontals = [paramsDict['h' + str(i)] for i in (1,2,3,4)]
        verticals = [paramsDict['v' + str(i)] for i in (1,2,3,4)]
        horizontalsColors = paramsDict['h_colors']
        verticalColors = paramsDict['v_colors']


        if len(img.shape) == 2:
            height,width= img.shape
        else:
            height,width,dimen= img.shape

        out = img.copy()

        for i,h in enumerate(horizontals):
            out[int(h/1000.0 * (height-1)) ,:] = horizontalsColors[i]
        for i,v in enumerate(verticals):
            out[:,int(v/1000.0 * (width-1))] = verticalColors[i]

        return out

class classifyLinesOperation(abstractFilterOperation):
    """ Classify rows and columns as one or zero. One stands for 'yes there is a mouse in this line', zero stands for
    'there is no mouse in this line'. The return type is a list containing two lists. The first with the classification of the
    rows and the second with classification of columns. Naturaly, the size of the first list should be img.shape[0] and
    the size of the second img.shape[0]. A line is split by 4 points l1,l2,l3 and l4. If a point x in the line is
    between l1 and l2(inclusive) or(inclisve or) between l3 and l4(inclusive) then x is of class one, else it is class
    zero. """
    def __init__(self):
        pass

    def apply(self, img, paramDict):
        horizontal = [paramDict['h' + str(i)] for i in (1,2,3,4)]
        verticals  = [paramDict['v' + str(i)] for i in (1,2,3,4)]

        height, width = img.shape[0], img.shape[1]
        h1,h2,h3,h4 = [int(h/1000.0 * (height-1)) for h in horizontal]
        v1,v2,v3,v4 = [int(v/1000.0 * (width-1) ) for v in verticals ]

        labels_rows = []
        labels_columns = []

        for row in range(img.shape[0]):
            labels_rows.append(1.0 if (((row <= h2) and (row >= h1)) or ((row <= h4) and (row >= h3))) else 0.0)

        for col in range(img.shape[1]):
            labels_columns.append(1.0 if (((col <= v2) and (col >= v1)) or ((col <= v4) and (col >= v3))) else 0.0)

        return labels_rows, labels_columns

class rangePainterOperation(abstractFilterOperation):
    def init(self):
        pass

    def apply(self, img, paramDict):
        rowLabels = paramDict['rowLabels']
        colLabels  = paramDict['colLabels']

        out = np.zeros_like(img)

        rowColor  = 100  if len(img.shape) == 2 else (255,0,0)
        colColor  = 100 if len(img.shape) == 2 else (0,0,255)


        for row in range(img.shape[0]):
            if rowLabels[row] == 1.0:
                out[row,:] += rowColor

        for col in range(img.shape[1]):
            if colLabels[col] == 1.0:
                out[:,col] += colColor

        return out





class getHSV_componentOperation(abstractFilterOperation):
    def __init__(self, component='h'):
        componentDict = {'h':0, 's':1, 'v':2}
        self.component = componentDict[component]

    def apply(self, img, paramsDict):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:,:,self.component]

class cannyOperation(abstractFilterOperation):
    defaultParamDict = {'threshold1':100, 'threshold2':150, 'apertureSize':3, 'L2gradient':True}
    def __init__(self):
        pass

    def apply(self, img, paramsDict = defaultParamDict):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(img,**paramsDict) 
    

class abstractMultiInputOperation(object):
    def __init__(self):
        pass
    
    def apply(self, inputList, paramsDictList):
        pass

class paint4Horizontals4VerticalsMultiInputOperation(abstractMultiInputOperation):
    def __init__(self):
        self.painter = paint4Horizontals4VerticalsOperation()

    def apply(self, inputList, paramsDictList):
        return [self.painter.apply(x, paramsDictList[i]) for i,x in enumerate(inputList)]

class convertGray2BGR_MultiInputOperation(abstractMultiInputOperation):
    def __init__(self):
        pass
    
    def apply(self, inputList, paramsDictList=[]):
        return [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in inputList]

class Sample(object):
    def __init__(self, vectorData, target):
        self.target = target # zero or one
        self.data = vectorData # unidimensional vector 

class FilterSampleCollection(object):
    def __init__(self, filterDescription='nonSpecified'):
        #each is a list of Sample objects(see class above)
        self.rowSamples = []
        self.colSamples = []
        self.filterDescription = filterDescription

    def addSamples(self, rowSamp, colSamp):
        """ each input is a list of Sample objects """
        self.rowSamples += rowSamp
        self.colSamples += colSamp

class classifyLinesMultiInputOperation(abstractMultiInputOperation):
    def __init__(self):
        self.lineClassifier = classifyLinesOperation()

    def apply(self, imgList, paramDict):
        """paramDict should contain all h's and v's and a list of sample collections. The sample collections should be
        ordered in tha same way the imgs are. The h's and v's will be sent to a classifyLinesOperation.
        
        The method will add samples to the already initialized collections. There is nothing being returned"""
        sampleCollections = paramDict['collectionList']
        for i,img in enumerate(imgList):
            rowLabels, colLabels = self.lineClassifier.apply(img, paramDict)
            rowSamples = [Sample(img[row,:], rowLabels[row]) for row in range(len(rowLabels))]
            colSamples = [Sample(img[:,col], colLabels[col]) for col in range(len(colLabels))]
            sampleCollections[i].addSamples(rowSamples, colSamples)

        return

class pyramidDownOperation(abstractFilterOperation):
    def __init__(self, n):
        self.n = n

    def apply(self, img, paramDict = dict()):
        out = img
        for k in range(self.n):
            out = cv2.pyrDown(out)
        return out

class pyramidDownMultiInputOperation(abstractMultiInputOperation):
    def __init__(self, n):
        self.pyramidDowOp = pyramidDownOperation(n)

    def apply(self, imgList, paramDict = dict()):
        return [self.pyramidDowOp.apply(x) for x in imgList]

class laplacianLvOperation(abstractFilterOperation):
    def __init__(self, n):
        self.n = n

    def apply(self, img, paramDict = dict()):
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for k in range(self.n):
            out = cv2.pyrDown(out)
        downUp = cv2.pyrUp(cv2.pyrDown(out))

        return out - downUp

class laplacianLvMultiInputOperation(abstractMultiInputOperation):
    def __init__(self, n):
        self.laplaceLvOp = laplacianLvOperation(n)

    def apply(self, imgList, paramDict = dict()):
        return [self.laplaceLvOp.apply(x) for x in imgList]
