import numpy as np
import cv2

class devika_cv:
    def __init__(self):
        pass

    @staticmethod
    def draw_flow(img, flow, step=8):
        """Assumes there is a flow for every point in the image. Draw flow on image at
        specified points. The points are: one every step points starting at step/2. The
        flow passed should have been calculated with calcOpticalFlowFarneback. That is,
        the flow is a displacement vector."""
        h, w = img.shape[:2]
        #compute points in which to draw flow
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
        #get flow at those points
        fx, fy = flow[y,x].T
        #write matrix containing lines of the type (x0,y0,x1,y1)
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        #round up to nears integer
        lines = np.int32(lines + 0.5)
        #make sure we're dealing with a BGR image
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #draw multiple lines
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        #draw circles 
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

        return vis

    @staticmethod
    def break_left_right(img):
      h, w = img.shape[:2]
      left = img[:,1:w/2-1]
      right = img[:,w/2:w]
      return left,right
