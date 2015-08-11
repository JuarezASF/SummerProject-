#!/usr/bin/python
import sys
sys.path.insert(0, './devikaFilterExperiment/')
import Tkinter as tk
from AbstractDemo import *
import cv2
import numpy as np
from DevikasFilterDemo import *
import pathlib

from GUI import RatFinderGUI


help = "\n\n HELP \n\n This program will launch a GUI and run devikas Filter to segment \
the floor and then track the mouse. \n\n"
print help

if __name__ == '__main__':
    root = tk.Tk()
    app = RatFinderGUI(root, guiUI_path = './')
    ratFinder = DevikasFilterDemo(app)
    app.setRatFinder(ratFinder)

    videoFiles = list(p for p in pathlib.Path('../video/mp4/').iterdir() if p.is_file() and p.name[0] != '.')
    for file in videoFiles:
        app.addVideo(file.as_posix(), file.name)

    
    root.after(1, app.executionLoop)
    root.mainloop()
