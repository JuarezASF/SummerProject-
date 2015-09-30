import cv2
import sys
sys.path.append('../')
sys.path.append('../flowExperiment')

import jasf
from jasf import jasf_cv
import  flowUtil
import numpy as np



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

    #initialize windows to be used
    jasf.cv.getManyWindows(['input', 'flow', 'flowDiff'])

    #make sure we have initial values for past things
    #at the first iteration they are required, thus we must initialize them to something
    currentFlow, oldFlow = np.zeros((0,0), dtype = np.uint8), np.zeros((0,0), dtype = np.uint8)
    allBlack = np.zeros_like(frame)
    prev, new, goodI = flowComputer.apply(frame, returnIndexes=True)

    while True:
        #keep data of previous flow computation
        oldPrev, oldNew, oldGoodI = prev, new, goodI

        #quit if 'q' is pressed
        ch = jasf.cv.waitKey(5)
        if ch == ord('q'):
            break

        #get new frame and stop if we're not able to read
        ret, frame = cam.read()
        if ret == False:
            break

        prev, new, goodI = flowComputer.apply(frame, returnIndexes=True)

        #those are the points for which the computation of the flow was sucessfull
        newGood = new[goodI]
        oldGood = prev[goodI]

        #this is the current flow on all points where we were able to fidn flow
        currentFlow = new - prev

        #if we have at least one inex valid
        if len(goodI) != 0 and  len(oldGoodI) != 0:
            #get indexes where both old flow and new flow computation worked
            matchingFlowPositions = np.intersect1d(goodI, oldGoodI)

            #get data of the current flow where we also have data for the previous flow
            matching_new= new[matchingFlowPositions]
            matching_old= prev[matchingFlowPositions]

            #this is the resulting flow(a subset of currentFlow)
            matching_currentFlow = matching_new - matching_old

            #get data of the old flow for points where we have knew flow data
            matching_oldNew = oldNew[matchingFlowPositions]
            matching_oldOld = oldPrev[matchingFlowPositions]

            matching_oldFlow = matching_oldNew - matching_oldOld

            #compute difference between flows
            flowDiff = (matching_currentFlow - matching_oldFlow)

            #we will draw the difference vector starting at each point. To do this we need:
            #every point where there is something to draw
            old_flowDiff = grid[matchingFlowPositions]
            #the end point of the arrow at each point where we're drawing something
            new_flowDiff = old_flowDiff + flowDiff

            #show result of flow difference computation
            showDiff = flowUtil.draw_flow(allBlack.copy(), old_flowDiff, new_flowDiff, th=0.5)
            cv2.imshow('flowDiff', showDiff)

        #draw current flow on all known points
        show = flowUtil.draw_flow(frame.copy(), oldGood, newGood, th=0.5)

        
        #show input frame and frame with flow arrows drawn
        cv2.imshow('input', frame)
        cv2.imshow('flow', show)


    cv2.destroyAllWindows()
    cam.release()
