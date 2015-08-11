import sys
sys.path.insert(0, '../')
import jasf
from jasf import jasf_cv
import cv2
import flowUtil

cam = cv2.VideoCapture(0)
ret, frame = cam.read()
h,w,d = frame.shape
print frame.shape

half_side = 200
grid = flowUtil.getGrid(w/2-half_side, h/2 - half_side, half_side*2, half_side*2, 5, 5)

flowComputer = flowUtil.FlowComputer()
flowComputer.setGrid(grid)

ret, frame = cam.read()
flowFilter = flowUtil.FlowFilter()
flowFilter2 = flowUtil.FlowFilter_ConnectedRegions(frame.shape[:2])

jasf_cv.getInputWindow()
jasf_cv.getSettingsWindow()

jasf.cv.setManyTrackbars(['low', 'upper'], [10, 25], [100, 100])
jasf.cv.setManyTrackbars(['low2', 'upper2'], [150, 50000], [10000, 50000])


while cam.isOpened():
    ch = cv2.waitKey(5) & 0xFFFF
    if ch == ord('q'):
        break
    
    ret, frame = cam.read()

    low_th, up_th = jasf.cv.readManyTrackbars(['low', 'upper'])
    low_th_area, up_th_area = jasf.cv.readManyTrackbars(['low2', 'upper2'])

    flowInput = jasf_cv.convertBGR2Gray(frame)
    oldP, newP = flowComputer.apply(flowInput)

    flowFilter.setTh(low_th, up_th)
    oldP, newP = flowFilter.apply(oldP, newP)

    flowFilter2.setTh(low_th_area, up_th_area)
    oldP, newP = flowFilter2.apply(oldP, newP, debugMode = True)


    flowImg = flowUtil.draw_flow(frame, oldP, newP, (255,0,0), 1, 1, 2, th = 0.2)

    cv2.imshow('input', frame)

cam.release()
cv2.destroyAllWindows()
