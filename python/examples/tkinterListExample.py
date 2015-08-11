from Tkinter import *

def dealWithIt(event):
  print 'DoubleClick Captured!'

master = Tk()

listbox = Listbox(master)
listbox.pack()

listbox.insert(END, "a list entry")

for item in ["one", "two", "three", "four"]:
    listbox.insert(END, item)
listbox.bind("<Double-Button-1>", dealWithIt)

mainloop()
