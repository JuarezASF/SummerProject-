import cv2
import sys
sys.path.append('../')
sys.path.append('../flowExperiment')

import jasf
from jasf import jasf_cv
import  flowUtil
import numpy as np


class FlowDiffComputer:

    def __init__(self, flowComputer):
        """ Flow computer must have all the information required to compute flow.
        I leave to you correctly initializing this.
        
        From the flowComputer Object expects it to:
            *provide a apply method with the returnIndexes=True option available
            *provide a public grid field
            """
        self.flowComputer = flowComputer
        #initialize required variables
        allZeros =  np.zeros((0,2), dtype = np.uint8)
        self.previousStartP, self.previousEndP, self.previousGoodI = allZeros.copy(), allZeros.copy(), allZeros.copy()
        self.currentStartP, self.currentEndP, self.currentGoodI = allZeros.copy(), allZeros.copy(), allZeros.copy()

    def apply(self, frame, returnIndo2Draw=False):
        """In case you want to plot the magnitude of the flow diff, use the following as an example:
            flowP, flowMag = diffComputer.apply(frame)
            output[flowP[:,1], flowP[:,0]] = flowMag

            flowP is array o shape(N,2): N entradas do tipo [y,x] <ambos np.uint64>
            flowMag is array o shape(N): N entradas do tipo real <tipo float>
            """
        #keep data of previous flow computation
        self.previousStartP, self.previousEndP, self.previousGoodI = self.currentStartP, self.currentEndP, self.currentGoodI

        #compute new flow[require flow to answer back with all points and the list of valid indexes]
        self.currentStartP, self.currentEndP, self.currentGoodI  = self.flowComputer.apply(frame, returnIndexes=True)


        #if we have at least one inex valid in both valid idexes(previous and current iteration)
        if len(self.currentGoodI) != 0 and  len(self.previousGoodI) != 0:
            #get indexes where both old flow and new flow computation worked
            matchingFlowPositions = np.intersect1d(self.previousGoodI, self.currentGoodI)

            #get data of the current flow where we also have data for the previous flow
            matching_currentStartP = self.currentStartP[matchingFlowPositions]
            matching_currentEndP   = self.currentEndP[matchingFlowPositions]

            #this is the flow for the points where there is flow for both iterations
            matching_currentFlow = matching_currentEndP - matching_currentStartP

            #get data of the old flow for points where we have new flow data
            matching_previousStartP = self.previousStartP[matchingFlowPositions]
            matching_previousEndP   = self.previousEndP[matchingFlowPositions]

            #this is the past flow for the points where there is flow for both iterations
            matching_previousFlow = matching_previousEndP - matching_previousStartP

            #compute difference between flows
            flowDiff = (matching_currentFlow - matching_previousFlow)

            #compute norm of every matching flow diff
            flowDiffNorm = np.linalg.norm(flowDiff, axis=1)

            #we will draw the difference vector starting at each point. To do this we need:
            #every point where there is something to draw
            flowDiffStartP = self.flowComputer.grid[matchingFlowPositions]
            if returnIndo2Draw:
                #the end point of the arrow at each point where we're drawing something
                flowDiffEndP = flowDiffStartP + flowDiff
                return flowDiffStartP.astype(np.uint64), flowDiffEndP
            else:
                return flowDiffStartP.astype(np.uint64), flowDiffNorm
        else:
            grid = self.flowComputer.grid.astype(np.uint64)
            if returnIndo2Draw:
                return grid, grid
            else:
                return grid, np.zeros(grid.shape[0])




if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    #get first frame so we set width and whigth of the flow computer
    ret, frame = cam.read()
    flowComputer = flowUtil.FlowComputer()
    width, height = frame.shape[1], frame.shape[0]
    grid = flowUtil.getGrid(5,5, width-5, height-5, 10,10) 
    flowComputer.setGrid(grid)
    #initialize flowComputer with first frame(at this point, we have only one image and no flow is computer)
    flowComputer.apply(frame)

    #initialize flow diff computer
    diffComputer = FlowDiffComputer(flowComputer)

    #initialize windows to be used
    jasf.cv.getManyWindows(['input',  'flowDiffMag'])

    allBlack = np.zeros((height, width), dtype=np.uint8)

    while True:
        #quit if 'q' is pressed
        ch = jasf.cv.waitKey(5)
        if ch == ord('q'):
            break

        #get new frame and stop if we're not able to read
        ret, frame = cam.read()
        if ret == False:
            break
        
        #get difference in flow from this to the previous flow
        flowP, flowMag = diffComputer.apply(frame)

        #paint a black frame with the flow mag in the points where there was flow to compare
        output = allBlack.copy()
        output[flowP[:,1], flowP[:,0]] = 10*flowMag

        #show input frame and frame with flow arrows drawn
        cv2.imshow('input', frame)
        cv2.imshow('flowDiffMag', output)

    cv2.destroyAllWindows()
    cam.release()
