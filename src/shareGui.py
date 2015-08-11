__author__ = 'hollowed'

'''
This module is for easier sharing of the current gui object, to avoid having to pass `self`
as a parameter to segment_test.start() from gui.XSDImageSegmentation.
Every module in this project can call this independently instead of accepting
the object as an argument.
'''

gui = None

def setGui(guiObj):

    global gui
    gui = guiObj

def getGui():

    global gui
    return gui


