import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in (1,2,3,4,5,6):
    name = '../img/edge' + str(i) + '.png' 
    t = cv2.equalizeHist(cv2.imread('../img/edge' + str(i) + '.png',0))
    name = '../img/edge' + str(i) + 'eq.png' 
    cv2.imwrite(name, t)

