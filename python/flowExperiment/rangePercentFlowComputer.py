import sys
sys.path.insert(0, '../')
import jasf
from jasf import jasf_cv
import flowUtil
import cv2
import numpy as np
import pdb

class RangePerCentFlowComputer(flowUtil.FlowComputer):
    def __init__(self, min=0.95, max=1.0):
        super(RangePerCentFlowComputer, self).__init__()
        self.setPercentageInterval(min, max)

    def setPercentageInterval(self, min, max):
        self.minP = min
        self.maxP = max

    def apply(self, img):
        #grap regular flow
        prev, new = super(RangePerCentFlowComputer, self).apply(img)
        #compute flow magnitudes as norm of the difference
        flow = new - prev
        flowNorm = np.linalg.norm(flow, axis = 1)
        #sort flow endpoints according to its norm and get those in the appropriate range
        sortingIndexes  = flowNorm.argsort()
        selectedIndexes = sortingIndexes[int(self.minP*sortingIndexes.size):int(self.maxP*sortingIndexes.size)]

        newP_selected, oldP_selected = new[selectedIndexes], prev[selectedIndexes]

        return oldP_selected, newP_selected

def averageFlow(prev, new):
    #average origin of flow:
    averageOrigin = np.average(prev.reshape(-1, 1, 2), axis = 0)
    #average end point of flow:
    averageEnd = np.average(new.reshape(-1, 1, 2), axis = 0)

    return averageOrigin, averageEnd


if __name__ == "__main__":
    print 'running test mode of RangePerCentFlowComputer...'
    cam = cv2.VideoCapture(0)

    unit = RangePerCentFlowComputer()

    h,w = jasf.cv.getVideoCaptureFrameHeightWidth(cam)
    unit.setGrid(flowUtil.getGrid(0,0, w, h, 10, 10))

    jasf.cv.getManyWindows(['input', 'output']) 

    ret,frame = cam.read()
    output = np.zeros_like(frame)

    while True:
        ch = cv2.waitKey(30) & 0xFFFF
        if ch == ord('q'):
            break
        ret, frame = cam.read()
        frame = jasf_cv.convertBGR2Gray(frame)

        old, new = unit.apply(frame)
        if old.size != 0:
            averageOld, averageNew = averageFlow(old, new)
            output = flowUtil.draw_flow(frame, old, new, drawArrows = True, lenghtOfArrayArm = 3)
            output = flowUtil.draw_flow(output, averageOld,averageNew, flowColor = jasf.cv.green, flowThickness=2,
                    drawArrows = True, lenghtOfArrayArm = 5)


        cv2.imshow('input', frame)
        cv2.imshow('output', output)


    cv2.destroyAllWindows()
    cam.release()
