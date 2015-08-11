import pathlib
import cv2
from operator import itemgetter
from datetime import datetime

p = pathlib.Path('../../video/mp4/')
allFiles = [x for x in p.iterdir() if x.is_file() and x.name[0] != '.' and '.mp4' in x.name and x.name[0:4] == '2014']

#remove the ending '.mp4' from file names
fileNames = [x.name[:len(x.name)-4] for x in allFiles]

#get only the h-m-s time stamp
fileNameTimes = [[int(xx) for xx in x[len(x)-8:].split('-')] for x in fileNames]
fileNameTimes = [x[0:] for x in fileNameTimes]

#group file data structures with their processed name
#this is done so we can order the first according to the second
fileContainer = [[allFiles[i], fileNameTimes[i]] for i in range(len(allFiles))]

#order first according to minutes
fileContainer.sort(key= lambda x: x[1][1])
#now order according to hour. This way the previous ordering is kept in case of ties
fileContainer.sort(key= lambda x: x[1][0])
#print the result so we take a look and check
print [(x[0].name, x[1]) for x in fileContainer]


#read input file with querries 
lines = [line.rstrip('\n') for line in open('./input2.txt', 'r')]

X = [x.split() for x in lines]

XX = [[x[0]] + x[1].split(':') for x in X]
#XX = ['r/g', [hour, minute]]
#XX[0][0] = 'r/g' of the first querry
#XX[0][q] = [hour, minute, second] of the second querry

print XX

#intialize window and trackbars
import sys
sys.path.insert(0, '../')
import jasf
from jasf import jasf_cv

jasf_cv.getInputWindow()

for querry in XX:
    color, h, m = querry
    #convert string to int
    h,m = int(h), int(m)
    querryTime = datetime(1,1,1,hour = h,minute = m, second = 0)
    i = 0
    fileTime = datetime.now()#this variable will be kept
    while i < len(fileContainer) - 1:
        fileTime = datetime(1, 1, 1, hour = fileContainer[i][1][0],   minute = fileContainer[i][1][1],   second = 0)
        nextTime = datetime(1, 1, 1, hour = fileContainer[i+1][1][0], minute = fileContainer[i+1][1][1], second = 0)
        if querryTime >= fileTime and querryTime <= nextTime:
            break
        i += 1
    chosenFile = fileContainer[i]
    print querry, 'is inside video', chosenFile[0].as_posix()

    control_return = 'continue'

    if querry[0] == 'g':
        cam = cv2.VideoCapture(chosenFile[0].as_posix())
        fps = int(cam.get(cv2.CAP_PROP_FPS) + 0.5)
        frameCount = cam.get(cv2.CAP_PROP_FRAME_COUNT)


        fileTime = datetime(1, 1, 1, hour = fileContainer[i][1][0],   minute = fileContainer[i][1][1],   second = fileContainer[i][1][2])

        marker_s = (querryTime - fileTime).total_seconds()
        marker_f = int(marker_s * fps)
        marker_start = max(0, marker_f - 5 *fps)
        marker_end = min(frameCount, marker_f + 65*fps)
        count = marker_end - marker_start

        cam.set(cv2.CAP_PROP_POS_FRAMES, marker_start)

        control_mode = 'run'

        originalFileName = chosenFile[0].name


        outputName = './output/' + originalFileName[:len(originalFileName)-12]
        outputName += '%02d-%02d-%02d.avi'%(querryTime.hour, querryTime.minute, querryTime.second)
        print 'file will be writte to', outputName
        height, width = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        originalFPS = cam.get(cv2.CAP_PROP_FPS)
        videoWriter = cv2.VideoWriter(outputName, cv2.VideoWriter_fourcc(*'XVID'), originalFPS, (width, height))

        while count > 0:
            ch = cv2.waitKey(1)
            if ch == ord('q'):
                print 'ending one video due to user input'
                break

            if ch == 27:#esc command
                control_return = 'terminate'
                break

            if control_mode == 'run':
                ret, frame = cam.read()
                count -= 1
                videoWriter.write(frame)

        if control_return == 'terminate':
            print 'ending execution earlier due to user input'
            break
        cam.release()
        videoWriter.release()

cv2.destroyAllWindows()
