""" this file contains the GUI to be used and reused on the rat finder reserach project.
The inferface has 9 windows displayed on 3x3 grid, a list that allows the user to select
the video he wants to read, a stop/play button, a menu with sliders(scales) for settings."""

import Tkinter as tk
import pygubu
from abstractRatFinder import *
import cv2
import numpy as np
import Image, ImageTk

from TkOpenCV import *

class RatFinderGUI:
    def __init__(self, master, guiUI_path='../'):
        self.guiUI_path = guiUI_path
        self.control_running = False
        self.control_video_changed = False

        self.master = master

        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui file
        builder.add_from_file(guiUI_path + 'GUI.ui')

        #3: Create the widget using a master as parent
        self.mainwindow = mainwindow = builder.get_object('mainwindow', master)
   
        #------------------------------------
        #add menu to settings
        #------------------------------------

        # create a toplevel menu
        menubar = tk.Menu(master)
        menubar.add_command(label="settings", command= lambda: self.onSettingsPressed())
        menubar.add_command(label="Quit!", command=self.master.quit)
        # display the menu
        master.config(menu=menubar)

        #1: Create a builder
        self.settings_builder = settings_builder = pygubu.Builder()

        #2: Load an ui file
        settings_builder.add_from_file(guiUI_path + 'settingsGUI.ui')

        #3: Create the widget using a master as parent
        self.settings_topLevel = settings_topLevel = settings_builder.get_object('settingsWindow', mainwindow)

        settings_topLevel.iconify()
        settings_topLevel.protocol("WM_DELETE_WINDOW", lambda: self.settings_topLevel.withdraw())
    
        self.settingsDict = {}
        self.scaleList = []
        for i in range(1,10):
            #obtain the img label from ui  
            self.scaleList.append(settings_builder.get_object('Scale_'+str(i), settings_topLevel))
        #------------------------------------
        #configure the list to select available videos
        #------------------------------------
        self.video_listbox = builder.get_object('videoList', mainwindow)

        self.videoDict = {"default":guiUI_path + "../video/mp4/myFavoriteVideo.mp4"}

        for item in self.videoDict.keys():
          self.video_listbox.insert(tk.END, item)

        self.video_listbox.bind("<Double-Button-1>", lambda e: self.onVideoSelectionChanged(e))

        self.videoFile = self.videoDict[self.video_listbox.get(0)]
        #------------------------------------
        #Configure the 3x3 image grid
        #------------------------------------
        defaultImg = cv2.imread(guiUI_path + 'myLogo2.png')
        g,b,r = cv2.split(defaultImg)
        img = cv2.merge((r,g,b))
        img = cv2.resize(img,(180,200))
        self.outputImgs = []
        self.imageLabels = []
        for i in range(1,10):
            #obtain the img label from ui  
            self.imageLabels.append(builder.get_object('img_'+str(i), mainwindow))
            #convert opncv img to ImagemTk
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im) 
            self.outputImgs.append(imgtk)
            #set image to default
            self.imageLabels[i-1].configure(image=imgtk)
        #------------------------------------
        #Configure Play Button
        #------------------------------------
        self.button_state = 'play'
        self.button_play_stop = builder.get_object('button_control', mainwindow)
        self.button_play_stop.configure(command = lambda: self.onButtonPress())
    
        #------------------------------------
        #Configure Reset Track Button
        #------------------------------------
        self.button_reset = builder.get_object('button_resetTrack', mainwindow)
        self.button_reset.configure(command = lambda: self.onResetTrackButtonPress())

        #------------------------------------
        #Configure Log panel
        #------------------------------------
        self.scroll_panel = builder.get_object('tkscrolledframe', mainwindow)
        self.log_text =  builder.get_object('log_text', self.scroll_panel)
        self.log('[log init]')

        #This flag is used to control the creation of a user click input window
        self.userInputWindow_initialized = False

    def clearSettings(self):
        self.settingsDict = {}

    def addSettingsPropertie(self, prop, min=0, max=255, current = 0, step=0):
        currentSize = len(self.settingsDict.keys())
        self.settingsDict[prop] = currentSize
        self.scaleList[currentSize].config(label=prop, to = max, from_ = min, tickinterval=step)
        self.setSettingsPropeitie(prop, current)

    def readSettingsPropertie(self, prop):
        if prop not in self.settingsDict.keys():
            raise Exception('proportie ' + prop + ' not found on settings')
        return self.scaleList[self.settingsDict[prop]].get()
        
    def readManySettingsPropertie(self, prop_list):
        settings = dict()
        for prop in prop_list:
            if prop not in self.settingsDict.keys():
                raise Exception('proportie ' + prop + ' not found on settings')
            settings[prop] = self.scaleList[self.settingsDict[prop]].get()
        return settings

    def setSettingsPropeitie(self, prop, val):
        if prop not in self.settingsDict.keys():
            raise Exception('proportie' + prop + 'not found on settings')
        return self.scaleList[self.settingsDict[prop]].set(val)

    def onSettingsPressed(self):
        self.settings_topLevel.deiconify()
        self.settings_topLevel.lift()

    def onVideoSelectionChanged(self, event):
        """This method will re initialize the rat Finder with the new video"""
        itemIndex =  self.video_listbox.curselection()
        itemName = self.video_listbox.get(itemIndex)
        self.videoFile = self.videoDict[itemName]

        self.ratFinder.initialize(self.videoFile)

        self.control_running = False
        self.onButtonPress()

    def onButtonPress(self):
        if self.button_state == 'play':
            self.button_state = 'stop'
            self.button_play_stop.configure(text = 'stop')
            self.control_running = True
        else:
            self.button_state = 'play'
            self.button_play_stop.configure(text = 'play')
            self.control_running = False

    def onResetTrackButtonPress(self):
        self.onButtonPress()
        self.ratFinder.askForUserInput()
        self.onButtonPress()

    def addVideo(self, file, title):
        self.videoDict[title] = file
        self.video_listbox.insert(tk.END, title)


    def setRatFinder(self, ratFinder):
        self.ratFinder = ratFinder
        ratFinder.initialize(self.videoFile)

    def start(self):
        Tkinter.mainloop()


    def setImg(self, img, index):
        img = cv2.resize(img,(180,200))
        if (len(img.shape) < 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im) 
        self.outputImgs[index] = imgtk
        #set image to default
        self.imageLabels[index].configure(image=imgtk)

    def executionLoop(self):
        if self.control_running:
            try:
                self.ratFinder.update()
            except VideoInvalid as E:
                print E.reason
                self.onButtonPress()

        if self.control_running == True:
          self.master.after(10, self.executionLoop)
        else:
          self.master.after(100, self.executionLoop)

    def log(self, msg):
        self.log_text.insert(tk.END, msg+'\n')
        self.log_text.yview_pickplace("end")

    def onUserInputWindowDoubleClick(self, event, dealer):
        #outputting x and y coords to console
        self.userClickInput_point[0] = event.x
        self.userClickInput_point[1] = event.y
        self.newUserClickReading = True
        dealer(event)

    def openUserClickWindow(self, dealer, title="click window"):
        """Way to use this:
            1. call openUserClickWindow
            2. set img with setUserClickWindowImg
            3. read input with readUserInputClick
            4. close window with destroyUserClickWindow
            """
        self.userClickInputWindow = tk.Toplevel(self.mainwindow)
        self.userClickInputWindow.wm_title(title)

        img = cv2.imread(self.guiUI_path + './openCV_logo.png')
        imgtk = TkOpenCV.getImgtkFromOpenCV(img)
        self.userInputImgTk = imgtk

        self.userClickInputLabel = tk.Label(self.userClickInputWindow, image=imgtk)

        self.userInputWindow_initialized = True

        self.userClickInput_point = [-1,-1]
        self.newUserClickReading = False
        self.userClickInput_done = False

        #mouseclick event
        self.userClickInputLabel.bind("<Button 1>",lambda e:self.onUserInputWindowDoubleClick(e, dealer))

        for item in self.settingsDict.items():
            val = item[1]
            self.scaleList[val].bind("<ButtonRelease-1>", dealer)
        
        return self.userClickInputWindow

    def readUserInputClick(self):
        if self.userInputWindow_initialized == False:
            raise Exception('User Input Click Window not initialized!')
        return self.userClickInput_point, self.newUserClickReading

    def setClickWasRead(self):
        self.newUserClickReading = False

    def setUserClick_Done(self):
        self.newUserClickReading = False
        self.userClickInput_done = True
        #mouseclick event
        self.userClickInputLabel.bind("<Button 1>",'')

        for item in self.settingsDict.items():
            val = item[1]
            self.scaleList[val].bind("<ButtonRelease-1>", '')
        self.destroyUserClickWindow()

    def destroyUserClickWindow(self):
        if self.userInputWindow_initialized == False:
            raise Exception('User Input Click Window not initialized!')
        self.userClickInputWindow.destroy()
        self.userInputWindow_initialized = False

    def setUserClickWindowImg(self, img):
        if self.userInputWindow_initialized == False:
            raise Exception('User Input Click Window not initialized!')
        imgtk = TkOpenCV.getImgtkFromOpenCV(img)
        self.userInputImgTk = imgtk
        self.userClickInputLabel.configure(image = imgtk)
        self.userClickInputLabel.pack(side="top", fill="both", expand=True, padx=0, pady=0)
