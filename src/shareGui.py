__author__ = 'hollowed'


#This module is for easy sharing of the current gui object, so other classes and modules can modify gui widgets.


gui = None

def setGui(guiObj):

    global gui
    gui = guiObj

def getGui():

    global gui
    return gui


