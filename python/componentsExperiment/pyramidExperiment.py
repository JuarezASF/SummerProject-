import sys
sys.path.insert(0, '../')
import cv2
from jasf import jasf_cv

from config import img2load
img = cv2.imread(img2load, 1)

print 'building pyramid down...'
show = [img]
input = img
for i in range(3):
    input = cv2.pyrDown(input.copy()) 
    show.append(input.copy())

windows = ['input']
windows += ['pyrDown' + str(i) for i in (1,2,3)]
sizes = [(240,320), (120,160), (60,80), (30,40)]
x_sum, y_sum = 0,0
for i,win in enumerate(windows):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, sizes[i][1], sizes[i][0])
    cv2.moveWindow(win, x_sum, y_sum)
    x_sum += sizes[i][1]
    y_sum += sizes[i][0]
    cv2.imshow(win, show[i])
    cv2.waitKey(0)
    cv2.imwrite('../../tex/meetingPresentation2/' + win + '.png', show[i])


print 'building pyramid up...'

pyrUp_img = []
for i in range(3):
    input = cv2.pyrUp(input.copy()) 
    pyrUp_img.append(input.copy())

windows = ['pyrUp' + str(i) for i in (1,2,3)]
sizes = [(240,320), (120,160), (60,80)]
sizes.reverse()
for i,win in enumerate(windows):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, sizes[i][1], sizes[i][0])
    x_sum += sizes[i][1]
    y_sum -= sizes[i][0]
    cv2.moveWindow(win, x_sum, y_sum)
    cv2.imshow(win, pyrUp_img[i])
    cv2.waitKey(0)
    cv2.imwrite('../../tex/meetingPresentation2/' + win + '.png', pyrUp_img[i])

cv2.waitKey(0)
cv2.destroyAllWindows()

print 'building laplacian pyramids...'


cv2.namedWindow('current', cv2.WINDOW_NORMAL)
cv2.namedWindow('down', cv2.WINDOW_NORMAL)
cv2.namedWindow('up', cv2.WINDOW_NORMAL)
cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
sizes = [(240,320), (120,160),(60,80)]
current = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for k in (0,1,2):
    down = cv2.pyrDown(current) 
    up = cv2.pyrUp(down)
    diff = current - up
    x_sum = 0
    for name in ('current', 'down', 'up', 'diff'):
        if name == 'down':
            cv2.resizeWindow(name, sizes[k][1]/2, sizes[k][0]/2)
        else:
            cv2.resizeWindow(name, sizes[k][1], sizes[k][0])
        cv2.moveWindow(name, x_sum, 0)
        x_sum += sizes[k][1]
    cv2.imshow('current', current)
    cv2.imshow('down', down)
    cv2.imshow('up', up)
    cv2.imshow('diff', diff)
    current = down
    cv2.waitKey(0)


print 'press again to quit....'
cv2.waitKey(0)

cv2.destroyAllWindows()
