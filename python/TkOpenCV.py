import Tkinter as tk
import cv2
import numpy as np
import Image, ImageTk


class TkOpenCV:
    def __init__(self):
        pass

    @staticmethod
    def getImgtkFromOpenCV(img):
        if (len(img.shape) < 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im) 
        return imgtk
