__author__ = 'hollowed'

#standard
import sys
import os
import subprocess
import platform
import time

#anaconda
from PIL import Image
from PyQt4 import QtGui as qt, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavBar
import numpy
import h5py

#here
import segmentation
import plotframe
import shareGui
import finalize
import output
import convertdata

#Hello, future developers! The main process for the program is located here. Most functions (in all modules) contain a breif description as the first line under their defenition,
#and explanatory comments throughout.

#--------------------------------------------------------------------------------------------------------------------------------------

class XSDImageSegmentation(qt.QMainWindow):

    def __init__(self, displaySize):
        # displaySize is of type QRect

        qt.QMainWindow.__init__(self)

        #Creating main app window
        self.WIDTH = 900
        self.HEIGHT = 600
        self.dispWidth = displaySize.width()
        self.dispHeight = displaySize.height()
        self.x = (self.dispWidth/2) - self.WIDTH/2
        self.y = (self.dispHeight/2) - self.HEIGHT/2
        self.setGeometry(self.x, self.y, self.WIDTH, self.HEIGHT)
        self.setWindowTitle("XSD Image Segmentation")
        self.setWindowIcon(qt.QIcon('icon.png'))
        self.channelsPopup = None
        self.exchangePopup = None

        self.isReset = True
        self.dataPaths = [] #paths to data arrays that will be used for segmentation
        self.imagePaths = [] #paths to images used for results display only
        #^^^if the user originally opened an image file for segmentation, rather than an hdf5,
        #there will be no difference between the dataPath and imagePath for that index of each list
        self.hdfIndex = 0 #current hdf being worked on (if multiple are selected for segmentation)
        self.imageNum = 0 #same as above, with consideration to other images as well
        self.hdfs = [] #opened hdf file objects
        self.hdfPaths = [] #path strings to opened hdf files
        self.hdfNames = [] #strings of filenames minus the rest of the path
        self.exchangeGroup = None #string of which exchange group is being used with the current hdf file (or batch of hdf files)
        self.axisInfo = []
        self.segmentPaths = []
        self.segmentData = [] #list of data returned from the segmentation process
        self.results = [] #list of data returned from the finishing process
        self.channels = [] #list of channels selected for the hdf file of the corresponding list index
        self.currentData = 0 #the current working file of all files selected for segmentation
        self.divideType = 0
        self.maxPixelDist = 0
        self.smoothingIterations = 0
        self.displayLog = True
        self.displayPlts = False
        self.displayRawData = False
        self.rawData = [] #pixel data values
        self.customParams = [] #array of boolean indicating whether or not the image at that index is still on the default parameters
        #below - self.parameterDict is a temporary dict that will change often, and is just used to gather info from the gui before
        #"saving" it at a specific index of self.paramters. After this is done, self.parameterDict can safely be changed elsewhere
        self.parameterDict = {'divideType': 2, 'maxPixelDist': 8, 'units':0, 'pixelEq': None, 'pixelStr':'',
                              'smoothingIterations': 0, 'haltThreshold':100, 'varThreshold':110, 'intThreshold':15}
        self.lockDict = False #the above dictionary cannot be modified when this is true
        self.parameters = [] #will become a list (of length # of images to be segmented), where each entry is another list,
                              #storing the chosen segmentation parameters for the image with the corresponding index in self.dataPaths

        self.frames()
        self.mainMenu()
        self.toolbar()
        self.buildMainFrame()
        self.buildLogFrame()
        self.buildResultsFrame()
        self.show()

#----------------------------------------CREATING MAIN LEVEL COMPONENTS----------------------------------------

    def frames(self):

        self.frameStack = qt.QStackedWidget(self)
        self.setCentralWidget(self.frameStack)
        frames = []

        #Creating different pages
        self.mainFrame = qt.QFrame(self)
        self.resultsFrame = qt.QFrame(self)
        self.logFrame = qt.QFrame(self)
        self.resultsFrame = qt.QFrame(self)
        frames.append(self.mainFrame)
        frames.append(self.resultsFrame)
        frames.append(self.logFrame)
        frames.append(self.resultsFrame)

        #Styling, and adding all fames to StackedWidget for easy switching
        for frame in frames:
            frame.setFrameStyle(qt.QFrame.Panel)
            frame.setFrameShadow(qt.QFrame.Sunken)
            frame.setLineWidth(3)
            self.frameStack.addWidget(frame)

        #Creating fonts
        self.header = qt.QFont('Consolas', 12, qt.QFont.Bold)
        self.emphasis1 = qt.QFont('Consolas', 10, qt.QFont.Bold)
        self.emphasis2 = qt.QFont('Consolas', 10)
        self.blue = qt.QPalette()
        self.blue.setColor(qt.QPalette.Foreground,QtCore.Qt.blue)
        self.red = qt.QPalette()
        self.red.setColor(qt.QPalette.Foreground,QtCore.Qt.red)
        return


    def mainMenu(self):

        #Creating menus
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        editMenu = mainMenu.addMenu('&Edit')
        viewMenu = mainMenu.addMenu('&View')
        helpMenu = mainMenu.addMenu('&Help')

        #Creating actions
        self.quit = qt.QAction("Exit", self)
        self.openResults = qt.QAction("Open Results from file", self)
        self.viewLog = qt.QAction("View activity log", self)
        self.copyLog = qt.QAction("Copy contents of activity Log", self)
        self.viewResults = qt.QAction("View results", self)

        #Configuring actions
        self.quit.triggered.connect(self.closeApplication)
        self.viewLog.triggered.connect(self.logView)
        self.copyLog.triggered.connect(self.copyLogContents)
        self.viewResults.triggered.connect(self.resultsView)
        self.openResults.triggered.connect(self.openResultsFromFile)
        self.quit.setShortcut("Ctrl+Q")
        self.openResults.setShortcut("Ctrl+O")
        self.viewLog.setShortcut("Ctrl+L")
        self.viewResults.setShortcut("Ctrl+R")

        #Populating menus
        fileMenu.addAction(self.openResults)
        fileMenu.addAction(self.quit)
        editMenu.addAction(self.copyLog)
        viewMenu.addAction(self.viewLog)
        viewMenu.addAction(self.viewResults)
        return


    def toolbar(self):

        #Creating toolbar
        self.toolBar = self.addToolBar('Toolbar')

        #Creating actions
        self.imageTool = qt.QAction(qt.QIcon('singleTool.png'), 'Import images for segmentation', self)
        self.logTool = qt.QAction(qt.QIcon('logTool.png'), 'View activity log', self)
        self.resultsTool = qt.QAction(qt.QIcon('resultsTool.png'), 'View results panel', self)

        #Assigning actions
        self.imageTool.triggered.connect(self.mainView)
        self.logTool.triggered.connect(self.logView)
        self.resultsTool.triggered.connect(self.resultsView)

        #Populating toolbar
        self.toolBar.addAction(self.imageTool)
        self.toolBar.addAction(self.logTool)
        self.toolBar.addAction(self.resultsTool)
        return

#----------------------------------------BUILDING ALL WINDOW VIEWS ---------------------------------------

    def buildMainFrame(self):

        #This frame displays the main UI, where images are imported, and parameters set

        #Creating frames/layouts to hold components
        centerFrame = qt.QFrame(self.mainFrame)
        fileButtonBox = qt.QFrame(self.mainFrame)
        filesBox = qt.QGroupBox('Files', centerFrame)
        rightFrame = qt.QFrame(centerFrame)
        outputBox = qt.QGroupBox('Output', rightFrame)
        bottomFrame = qt.QFrame(self.mainFrame)
        self.paramsBox = qt.QGroupBox('Parameters', rightFrame)

        mainLayout = qt.QVBoxLayout(self.mainFrame) #main frame layout
        centerLayout = qt.QHBoxLayout(centerFrame)
        openButtonsLayout = qt.QHBoxLayout(fileButtonBox)
        filesLayout = qt.QVBoxLayout(filesBox)
        leftLayout = qt.QVBoxLayout(rightFrame)
        paramsBoxLayout = qt.QVBoxLayout(self.paramsBox)
        paramsLayout = qt.QGridLayout()
        outputLayout = qt.QGridLayout(outputBox)
        bottomLayout = qt.QHBoxLayout(bottomFrame)

        #Creating components
        bottomFrame.setMaximumHeight(40)
        self.mainTitle = qt.QLabel('XSD Image Segmentation', self.mainFrame)
        self.mainTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.openImgButton = qt.QPushButton('Open images...', filesBox)
        self.openHDF5Button = qt.QPushButton('Open HDF5...', filesBox)
        self.fileLabel = qt.QLabel('Filenames:', filesBox)
        self.fileScroll = qt.QScrollArea(filesBox)
        self.fileList = qt.QListWidget(self.fileScroll)
        self.currentFileLabel = qt.QLabel('No file selected', self.paramsBox)
        self.fileSettingsLabel = qt.QLabel('Default settings', self.paramsBox)
        self.divideTypePrompt = qt.QLabel('Divide Type:', self.paramsBox)
        self.divideTypeCombo = qt.QComboBox(self.paramsBox)
        self.maxPixelDistPrompt = qt.QLabel('Maximum Pixel Distance:', self.paramsBox)
        self.maxPixelDistSpin = qt.QSpinBox(self.paramsBox)
        self.smoothCheck = qt.QCheckBox('Smooth images', self.paramsBox)
        self.smoothingIterationsPrompt = qt.QLabel('Smoothing Iterations:', self.paramsBox)
        self.smoothingIterationsSpin = qt.QSpinBox(self.paramsBox)
        self.haltThresholdPrompt = qt.QLabel('Halting Threshold:', self.paramsBox)
        self.haltThresholdSpin = qt.QSpinBox(self.paramsBox)
        self.unitsCombo = qt.QComboBox(self.paramsBox)
        self.pixelEqLabel = qt.QLabel('Easter Egg!', self.paramsBox)
        self.backgroundPrompt = qt.QLabel('Background detection:')
        self.bgVarianceLabel = qt.QLabel('Variance threshold:')
        self.bgVarianceSpin = qt.QSpinBox(self.paramsBox)
        self.bgIntensityLabel = qt.QLabel('Intensity threshold:')
        self.bgIntensitySpin = qt.QSpinBox(self.paramsBox)
        self.saveParamsBtn1 =qt.QPushButton('Save for all images', self.paramsBox)
        self.saveParamsBtn2 =qt.QPushButton('Save for all subsequent images', self.paramsBox)
        self.logCheck = qt.QCheckBox('Display activity log', outputBox)
        self.plotsCheck = qt.QCheckBox('Display Plots', outputBox)
        self.rawDataCheck = qt.QCheckBox('Display raw image data', outputBox)
        self.segment = qt.QPushButton('Segment', bottomFrame)
        self.segment.setStyleSheet('background-color: lightgreen')
        self.reset = qt.QPushButton('Reset', bottomFrame)
        self.reset.setStyleSheet('background-color: lightblue')
        self.progress = qt.QProgressBar(bottomFrame)

        #Configuring components
        self.openImgButton.clicked.connect(self.openImage)
        self.openHDF5Button.clicked.connect(self.openHDF5)
        self.reset.clicked.connect(self.resetAll)
        self.segment.clicked.connect(self.runSegmentation)
        self.smoothCheck.toggled.connect(self.toggleSmoothing)
        self.saveParamsBtn1.clicked.connect(self.saveAllParams)
        self.saveParamsBtn2.clicked.connect(self.saveAllSubParams)
        self.fileList.itemClicked.connect(self.changeParamsApperance)
        self.maxPixelDistSpin.setMinimum(2)
        self.smoothingIterationsSpin.setMinimum(0)
        self.mainTitle.setMaximumHeight(40)
        self.mainTitle.setFont(self.header)
        self.fileLabel.setFont(self.emphasis2)
        self.currentFileLabel.setFont(self.emphasis2)
        self.fileSettingsLabel.setFont(self.emphasis2)
        self.fileSettingsLabel.setPalette(self.red)
        self.pixelEqLabel.setText('')
        self.fileLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.fileScroll.setWidgetResizable(True)
        self.fileScroll.setWidget(self.fileList)
        self.divideTypeCombo.addItems(['0','1','2'])
        self.smoothingIterationsPrompt.setDisabled(True)
        self.smoothingIterationsSpin.setDisabled(True)
        self.haltThresholdSpin.setMinimum(1)
        self.haltThresholdSpin.setMaximum(999999)
        self.bgVarianceSpin.setMaximum(999999)
        self.bgIntensitySpin.setMaximum(999999)
        self.unitsCombo.addItems(['px', 'mm^2', 'um^2', 'nm^2'])
        self.unitsCombo.setDisabled(True)
        self.progress.setValue(0)
        self.progress.setDisabled(True)
        self.logCheck.setChecked(True)
        self.plotsCheck.setChecked(False)
        self.rawDataCheck.setChecked(False)
        self.paramsBox.setDisabled(True)

        #default values (change if needed)
        self.haltThresholdSpin.setValue(100) #once a segment is at this size (pixel area), do not cut it further
        self.maxPixelDistSpin.setValue(8) #if pixels i and j are this distance away, weight_ij = 0
        self.divideTypeCombo.setCurrentIndex(2) #how cut is determined
        self.smoothCheck.setChecked(False) #if image should be smoothed before segmenting
        self.smoothingIterationsSpin.setValue(0) #how many times to run through smoothing function
        self.bgVarianceSpin.setValue(100) #maximum variance for a segment to be considered background
        self.bgIntensitySpin.setValue(15) #maximum intensity for a segment to be considered background

        #save parameters on value change
        self.smoothCheck.toggled.connect(self.saveParams)
        self.haltThresholdSpin.valueChanged.connect(self.saveParams)
        self.haltThresholdSpin.valueChanged.connect(self.updateUnits)
        self.smoothingIterationsSpin.valueChanged.connect(self.saveParams)
        self.divideTypeCombo.activated.connect(self.saveParams)
        self.maxPixelDistSpin.valueChanged.connect(self.saveParams)
        self.bgVarianceSpin.valueChanged.connect(self.saveParams)
        self.bgIntensitySpin.valueChanged.connect(self.saveParams)
        self.unitsCombo.activated.connect(self.saveParams)
        self.unitsCombo.activated.connect(self.updateUnits)

        #Packing components
        openButtonsLayout.addWidget(self.openImgButton)
        openButtonsLayout.addWidget(self.openHDF5Button)
        filesLayout.addWidget(fileButtonBox)
        filesLayout.addWidget(self.fileLabel)
        filesLayout.addWidget(self.fileScroll)
        paramsLayout.addWidget(self.divideTypePrompt, 0, 0)
        paramsLayout.addWidget(self.divideTypeCombo, 0, 1)
        paramsLayout.addWidget(self.maxPixelDistPrompt, 1, 0)
        paramsLayout.addWidget(self.maxPixelDistSpin, 1, 1)
        paramsLayout.addWidget(self.haltThresholdPrompt, 2, 0)
        paramsLayout.addWidget(self.haltThresholdSpin, 2, 1)
        paramsLayout.addWidget(self.unitsCombo, 2, 2)
        paramsLayout.addWidget(self.pixelEqLabel, 3, 1)
        paramsLayout.addWidget(self.smoothCheck, 4, 0)
        paramsLayout.addWidget(self.smoothingIterationsPrompt, 4, 1)
        paramsLayout.addWidget(self.smoothingIterationsSpin, 5, 1)
        paramsLayout.addWidget(self.backgroundPrompt, 6, 0)
        paramsLayout.addWidget(self.bgVarianceLabel, 7, 0)
        paramsLayout.addWidget(self.bgVarianceSpin, 7, 1)
        paramsLayout.addWidget(self.bgIntensityLabel, 8, 0)
        paramsLayout.addWidget(self.bgIntensitySpin, 8, 1)
        paramsBoxLayout.addWidget(self.currentFileLabel)
        paramsBoxLayout.addWidget(self.fileSettingsLabel)
        paramsBoxLayout.addLayout(paramsLayout)
        paramsBoxLayout.addWidget(self.saveParamsBtn1)
        paramsBoxLayout.addWidget(self.saveParamsBtn2)
        outputLayout.addWidget(self.logCheck, 1, 0)
        outputLayout.addWidget(self.plotsCheck, 2,0)
        outputLayout.addWidget(self.rawDataCheck, 3, 0)
        leftLayout.addWidget(self.paramsBox)
        leftLayout.addWidget(outputBox)
        centerLayout.addWidget(filesBox)
        centerLayout.addWidget(rightFrame)
        bottomLayout.addWidget(self.reset)
        bottomLayout.addWidget(self.segment)
        bottomLayout.addWidget(self.progress)
        mainLayout.addWidget(self.mainTitle)
        mainLayout.addWidget(centerFrame)
        mainLayout.addWidget(bottomFrame)
        return


    def buildLogFrame(self):

        #This frame displays a log of the current status of the algorithms progress

        #Creating frames/layouts
        logProgressFrame = qt.QFrame(self.logFrame)

        logLayout = qt.QVBoxLayout(self.logFrame)
        logProgressLayout = qt.QHBoxLayout(logProgressFrame)

        #Creating components
        self.logProgress = qt.QProgressBar(logProgressFrame)
        self.log = qt.QPlainTextEdit(self.logFrame)
        self.logTitle = qt.QLabel('Activity Log', self.logFrame)
        self.logReset = qt.QPushButton('Reset', logProgressFrame)

        #Configuring components
        self.log.setReadOnly(True)
        self.logProgress.setValue(0)
        self.logTitle.setFont(self.header)
        self.logTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.logReset.clicked.connect(self.resetAll)
        self.logReset.setStyleSheet('background-color: lightblue')

        #Packing components
        logProgressLayout.addWidget(self.logReset)
        logProgressLayout.addWidget(self.logProgress)
        logLayout.addWidget(self.logTitle)
        logLayout.addWidget(self.log)
        logLayout.addWidget(logProgressFrame)
        return


    def buildResultsFrame(self):

        #This frame displays the results from the segmentation, including all segments as seperate images,
        #and scatter plots + histograms of the image

        #Creating frames/layouts
        self.resultTabs = qt.QTabWidget(self.resultsFrame)
        self.scatterTab = qt.QFrame(self.resultTabs)
        self.histogramTab = qt.QFrame(self.resultTabs)
        self.segmentsTab = qt.QFrame(self.resultTabs)
        self.rawDataTab = qt.QFrame(self.resultTabs)
        self.scatterScroll = qt.QScrollArea(self.scatterTab)
        self.histogramScroll = qt.QScrollArea(self.histogramTab)
        self.segmentsScroll = qt.QScrollArea(self.segmentsTab)
        self.scatterFrame = qt.QFrame(self.scatterScroll)
        self.histogramFrame = qt.QFrame(self.histogramScroll)
        self.segmentsFrame = qt.QFrame(self.segmentsScroll)
        self.bottomResultsFrame = qt.QFrame(self.resultsFrame)

        resultsLayout = qt.QVBoxLayout(self.resultsFrame)
        self.scatterTabLayout = qt.QVBoxLayout(self.scatterTab)
        self.scatterLayout = qt.QGridLayout(self.scatterFrame)
        self.histogramTabLayout = qt.QVBoxLayout(self.histogramTab)
        self.histogramLayout = qt.QGridLayout(self.histogramFrame)
        self.segmentsTabLayout = qt.QVBoxLayout(self.segmentsTab)
        self.segmentsLayout = qt.QGridLayout(self.segmentsFrame)
        self.bottomResultsLayout = qt.QHBoxLayout(self.bottomResultsFrame)
        self.rawDataLayout = qt.QVBoxLayout(self.rawDataTab)

        #Creating components
        self.resultsTitle = qt.QLabel('Results', self.resultsFrame)
        self.noResults = qt.QLabel('Run segmentation or open results from file to display.', self.segmentsFrame)
        self.resultsReset = qt.QPushButton('Reset', self.bottomResultsFrame)
        self.openSegmentDir = qt.QPushButton('Open segment location', self.bottomResultsFrame)
        self.resultsSpacer = qt.QLabel(' ', self.bottomResultsFrame)
        self.scatterAxesLabels = qt.QLabel('x-axis = pixel\ny-axis = second eigenvector', self.scatterFrame)
        self.histogramAxesLabels = qt.QLabel('x-axis = \ny-axis = second eigenvector', self.histogramFrame)
        self.rawDataDisplay = qt.QPlainTextEdit(self.rawDataTab)

        #Configuring components
        self.resultTabs.setTabEnabled(1, False)
        self.resultTabs.setTabEnabled(2, False)
        self.resultTabs.setTabEnabled(3, False)
        self.resultsTitle.setFont(self.header)
        self.resultsTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.noResults.setAlignment(QtCore.Qt.AlignCenter)
        self.resultsReset.clicked.connect(self.resetAll)
        self.resultsReset.setStyleSheet('background-color: lightblue')
        self.resultsReset.setMaximumWidth(80)
        self.openSegmentDir.clicked.connect(self.openSegmentDirectory)
        self.openSegmentDir.setDisabled(True)
        self.openSegmentDir.setMaximumWidth(160)
        self.scatterScroll.setWidget(self.scatterFrame)
        self.histogramScroll.setWidget(self.histogramFrame)
        self.segmentsScroll.setWidget(self.segmentsFrame)
        self.scatterAxesLabels.setFont(self.emphasis1)
        self.histogramAxesLabels.setFont(self.emphasis1)
        self.rawDataDisplay.setReadOnly(True)
        #To make scrollbars behave appropriately:
        scrolls = [self.segmentsScroll, self.scatterScroll, self.histogramScroll]
        for scroll in scrolls:
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            scroll.setWidgetResizable(True)

        #Packing components
        self.scatterTabLayout.addWidget(self.scatterScroll)
        self.histogramTabLayout.addWidget(self.histogramScroll)
        self.segmentsTabLayout.addWidget(self.segmentsScroll)
        self.segmentsLayout.addWidget(self.noResults, 0, 0)
        self.scatterLayout.addWidget(self.scatterAxesLabels, 0, 0)
        self.histogramLayout.addWidget(self.histogramAxesLabels, 0, 0)
        self.bottomResultsLayout.addWidget(self.resultsReset)
        self.bottomResultsLayout.addWidget(self.openSegmentDir)
        self.bottomResultsLayout.addWidget(self.resultsSpacer)
        self.rawDataLayout.addWidget(self.rawDataDisplay)
        self.resultTabs.addTab(self.segmentsTab, 'Segmentation')
        self.resultTabs.addTab(self.scatterTab, 'Scatter Plots')
        self.resultTabs.addTab(self.histogramTab, 'Histograms')
        self.resultTabs.addTab(self.rawDataTab, 'Raw Image Data')
        resultsLayout.addWidget(self.resultsTitle)
        resultsLayout.addWidget(self.resultTabs)
        resultsLayout.addWidget(self.bottomResultsFrame)
        return

#----------------------------------------TOOLBAR BUTTON FUNCTIONS----------------------------------------

    def mainView(self):

        #Brings the main frame back into view
        self.frameStack.setCurrentWidget(self.mainFrame)
        if(len(self.dataPaths) == 0):
            self.fileLabel.setFont(self.emphasis2)
        else:
            self.fileLabel.setFont(self.emphasis1)
        return


    def logView(self):
        #Brings the activity log into view
        self.frameStack.setCurrentWidget(self.logFrame)
        return


    def resultsView(self):
        #Brigns the results page into view
        self.frameStack.setCurrentWidget(self.resultsFrame)
        return


#---------------------------------------- WIDGET ACTION FUNCTIONS ----------------------------------------

    def resetAll(self):

        #Resets everything in the gui to its initial state

        self.lockDict = True #prevent changes made below from calling saveParams()

        self.enableAll()
        self.exchangePopup = None
        self.channelsPopup= None

        self.divideType = 0
        self.maxPixelDist = 0
        self.hdfIndex = 0
        self.imageNum = 0
        self.dataPaths = []
        self.imagePaths = []
        self.hdfs = []
        self.hdfPaths = []
        self.hdfNames = []
        self.axisInfo = []
        self.exchangeGroup = None
        self.segmentPaths = []
        self.segmentData = []
        self.results = []
        self.currentData = 0
        self.smoothingIterations = 0
        self.displayLog = True
        self.displayPlts = False
        self.displayRawData = False
        self.resultTabs.setTabEnabled(1, False)
        self.resultTabs.setTabEnabled(2, False)
        self.resultTabs.setTabEnabled(3, False)
        self.paramsBox.setDisabled(True)
        self.parameterDict = {'divideType': 2, 'maxPixelDist': 8, 'units':0, 'pixelEq': None, 'pixelStr':'',
                              'smoothingIterations': 0, 'haltThreshold':100, 'varThreshold':110, 'intThreshold':15}
        self.parameters = []
        self.customParams = []

        #defaults
        self.haltThresholdSpin.setValue(100)
        self.maxPixelDistSpin.setValue(8)
        self.divideTypeCombo.setCurrentIndex(2)
        self.smoothCheck.setChecked(False)
        self.smoothingIterationsSpin.setValue(0)
        self.bgVarianceSpin.setValue(100)
        self.bgIntensitySpin.setValue(15)

        self.fileSettingsLabel.setText('Default settings')
        self.currentFileLabel.setText('No file selected')
        self.unitsCombo.setCurrentIndex(0)
        self.unitsCombo.setDisabled(True)
        self.logCheck.setChecked(True)
        self.plotsCheck.setChecked(False)
        self.rawDataCheck.setChecked(False)
        self.smoothingIterationsPrompt.setDisabled(True)
        self.smoothingIterationsSpin.setDisabled(True)
        self.pixelEqLabel.setText('')
        self.progress.setValue(0)
        self.progress.setDisabled(True)
        self.logProgress.setValue(0)
        self.logProgress.setDisabled(True)
        self.log.clear()
        self.fileList.clear()
        self.fileLabel.setFont(self.emphasis2)
        self.fileSettingsLabel.setPalette(self.red)
        self.openSegmentDir.setDisabled(True)

        #This loop clears all images and plots from the results tabs
        layouts = [self.scatterLayout, self.histogramLayout, self.segmentsLayout]
        for layout in layouts:
            for i in reversed(range(layout.count())):
                #The reversed range is necessary to prevent error
                layout.itemAt(i).widget().setParent(None)

        #calls the clearAll() method in plotframe.py so that every plot is erased
        plotframe.clearAll()
        self.segmentsLayout.addWidget(self.noResults, 0, 0)
        self.scatterLayout.addWidget(self.scatterAxesLabels, 0, 0)
        self.histogramLayout.addWidget(self.histogramAxesLabels, 0, 0)
        self.noResults.show()
        self.resultTabs.setTabEnabled(1, False)
        self.resultTabs.setTabEnabled(2, False)
        self.resultTabs.setTabEnabled(3, False)

        self.lockDict = False
        self.isReset = True
        return


    def toggleSmoothing(self):

        #self.lockDict = True #to keep the params dictionaries from updating duuring this

        if(self.smoothCheck.isChecked()):
            self.smoothingIterationsSpin.setDisabled(False)
            self.smoothingIterationsPrompt.setDisabled(False)
            self.smoothingIterations = 1
            self.smoothingIterationsSpin.setValue(1)
        else:
            self.smoothingIterationsSpin.setDisabled(True)
            self.smoothingIterationsPrompt.setDisabled(True)
            self.smoothingIterations = 0
            self.smoothingIterationsSpin.setValue(0)

        #self.lockDict = False
        return


    def openImage(self):

        #this function opens image files

        if(self.isReset == False):
            answer = self.showMessage('Error', 'Window must be reset before segmenting again.\nReset Now?', 'question')
            if answer == 0:
                self.resetAll()
                time.sleep(1)
                self.openImage()
            return

        #open the images from file to be segmented
        temp = qt.QFileDialog.getOpenFileNames(self, 'Open Image(s)', QtCore.QDir.currentPath(),
                                               filter="Images (*.png *.bmp *.jpg *.jpeg *.tiff)")

        for file in temp:
            if(str(file) != ''):
                self.dataPaths.append(str(file))
                self.imagePaths.append(str(file))
                self.fileList.addItem(os.path.split(str(file))[-1])
                self.toDefaultParams() #reset params to default to avoid copying last opened image settings (could be removed)
                self.parameters.append(self.parameterDict.copy()) #add a default parameter dict to the list of image parameters
                self.customParams.append(False) #Add a 'False' to customParams since this image is still set to the defaults
                self.axisInfo.append(None) #No axis info for a non-hdf image
                self.imageNum += 1
            else:
                self.showMessage('Error', 'No files selected', 'warning')

        return


    def openHDF5(self):

        #written in reference to SimpleView2.py - Hong
        #this function imports HDF5 files

        self.hdfs = []
        self.hdfIndex = 0

        if(self.isReset == False):
            answer = self.showMessage('Error', 'Window must be reset before segmenting again.\nReset Now?', 'question')
            if answer == 0:
                self.resetAll()
                time.sleep(1)
                self.openHDF5()
            return

        #open the images from file to be segmented
        temp = qt.QFileDialog.getOpenFileNames(self, 'Open HDF5(s)', QtCore.QDir.currentPath(), filter="h5 (*.h5)")

        for file in temp:
            if(str(file) != ''):
                self.hdfPaths.append(str(file))
                self.hdfNames.append(os.path.split(str(file))[-1])
            else:
                self.showMessage('Error', 'No files selected', 'warning')

        self.exchangeGroup = None
        self.invalidFiles = []

        #open each selected HDF5 with h5py, append each file to a list of files
        for file in self.hdfPaths:
            f = h5py.File(os.path.abspath(file),"r")
            self.hdfs.append(f)

        #open the exchange popup for the first file in self.hdfs (popup to choose which exchange directory to look for data in)
        try:
            self.exchangePopup = ExchangePopup(self.hdfs[0], self.hdfNames[0], len(self.hdfs))
            self.exchangePopup.show()
        except IndexError:
            #self.hdfs is empty
            return
        return


    def selectChannels(self, exchange):

        #written in refernce to SimpleView2.py - Hong
        #this function handles prompting the user for which HDF5 channels should be included in segmentation

        #just hide the exchange popup for now until we know user is not going to use it again
        self.exchangePopup.hide()
        self.exchangeGroup = exchange
        dataStr = '{}/data'.format(exchange)
        n = self.hdfIndex

        #-------------------- check for XRF data in selected exchnage group --------------------
        try:
            data = self.hdfs[n][dataStr] #look for XRF data
            self.exchangePopup.close()
        except KeyError:
            #group had no 'data' key
            answer = self.showMessage('Error', 'This exchange group has no data. \nSelect new exchange? \n(Press no to remove this HDF file from the queue)', 'question')
            if answer == 0:
                self.exchangePopup.show() #re-show popup so user can reselect exchange
                return
            elif answer == 1:
                #user decided to remove this file from the queue, get rid of its info and close the popup
                self.exchangePopup.close()
                self.hdfs[n].close()
                del self.hdfs[n]
                del self.hdfPaths[n]
                del self.hdfNames[n]
                del self.hdfs[n]
                try:
                    #open new popup for next hdf5
                    self.exchangePopup = ExchangePopup(self.hdfs[n+1], self.hdfNames[n+1], len(self.hdfs))
                    self.hdfIndex += 1
                    self.exchangePopup.show()
                    return
                except IndexError:
                    #in case there are no more hdf files to open a popup for (or only one hdf was ever selected)
                    return
            return

        #-------------------- check for channel data in selected exchange group --------------------
        try:
            #look for channel data
            channelStr = '{}/channel_names'.format(exchange)
            channels = self.hdfs[n][channelStr]
            self.exchangePopup.close()
        except KeyError:
            #group had no channel_names key
            answer = self.showMessage('Error', 'This exchange group has no channel data.\nSelect new exchange? \n(Press no to remove this HDF file from the queue)', 'question')
            if answer == 0:
                self.exchangePopup.show()
                return
            elif answer == 1:
                self.exchangePopup.close()
                self.hdfs[n].close()
                del self.hdfs[n]
                del self.hdfPaths[n]
                del self.hdfNames[n]
                del self.hdfs[n]
                try:
                    self.exchangePopup = ExchangePopup(self.hdfs[n+1].keys(), self.hdfNames[n+1], len(self.hdfs))
                    self.hdfIndex += 1
                    self.exchangePopup.show()
                    return
                except IndexError:
                    return
            return

        #spawn a chanels popup, for the user to choose channels to include in segmentation
        self.channelsPopup = ChannelsPopup(channels)
        self.channelsPopup.show()
        return


    def setChannels(self, selectedChannels, selectedIndices, stack):

        #this function prints the selected files and channels to the gui, and calls convertdata

        self.channelsPopup.close()

        #Adding filenames to the gui file list, and calling convertdata.toArray() to retrieve the desired channel data from
        #the hdf file. If stack == True, each channel's pixel values will be "stacked" ontop of the others, making a multi-layered
        #dataset. Otherwise, each channel is treated as a seperate image. All resultant data arrays are saved to numpy files for
        # segmentation and also to image files for results display. Returned is the path the the numpy files(s), and path to the images file(s)

        for n in range(len(self.hdfs)):
            try:
                if stack:
                    self.axisInfo.append(self.getAxisInfo(n)) #find info on units and real pixel size
                    newData, newImages = convertdata.toStackedArray(self.hdfPaths[n], self.hdfs[n], self.exchangeGroup, selectedIndices, selectedChannels)
                    self.fileList.addItem('{} - {} - ({})'.format(self.hdfNames[n], self.exchangeGroup, selectedChannels))
                    self.dataPaths.append(newData)
                    self.imagePaths.append(newImages)
                    self.toDefaultParams() #reset params to default to avoid copying last opened image settings (could be removed)
                    self.parameters.append(self.parameterDict.copy()) #add a default parameter dict to the list of image parameters
                    if self.axisInfo[n] == None: self.parameters[self.imageNum]['pixelStr'] = 'No unit data found' #Display this string if getAxisInfo returns None (and is an hdf5)
                    self.customParams.append(False) #Add a 'False' to customParams since this image is still setto the defaults
                    self.hdfIndex += 1
                    self.imageNum += 1
                else:
                    axisInfo = self.getAxisInfo(n)
                    newData, newImages = convertdata.toArray(self.hdfPaths[n], self.hdfs[n], self.exchangeGroup, selectedIndices, selectedChannels)
                    for channel in selectedChannels:
                        self.fileList.addItem('{} - {} - ({})'.format(self.hdfNames[n], self.exchangeGroup, channel))
                        self.toDefaultParams()
                        self.parameters.append(self.parameterDict.copy())
                        self.axisInfo.append(axisInfo)
                        if axisInfo == None: self.parameters[self.imageNum]['pixelStr'] = 'No unit data found'
                        self.customParams.append(False)
                        self.imageNum += 1
                    self.dataPaths.extend(newData)
                    self.imagePaths.extend(newImages)
                    self.hdfIndex += 1

                self.hdfs[n].close()
            except KeyError, ValueError:
                self.invalidFiles.append(n)

        if len(self.invalidFiles) == 0:
            return

        names = ''
        for index in self.invalidFiles:
            names = '{}{}\n'.format(names, self.hdfNames[index])
        self.showMessage('Errors', 'The following files raised errors with the chosen exchange/channel settings and were removed from the queue:\n\n{}'.format(names), 'message')
        return


    def getAxisInfo(self, n):

        #this function gets the units used for the axis of the current hdf file

        #----------get units----------

        hdf = self.hdfs[n]
        key = self.exchangeGroup
        try: units = hdf[key]['x_axis'].attrs.get('units') #assumes this exchange group has an 'x_axis' dataset,
        except KeyError, ValueError: return None           #and that dataset has a 'units' attribute in a format like ('units', 'mm')
        if units == None:
            return None

        #scale to standard SI units (meters) for standarized conversion later
        #add more 'if' statements here if some units are missing
        if units == 'mm': siScaling = 10**-3
        elif units == 'um' or units == 'microns': siScaling = 10**-6
        elif units == 'nm': siScaling = 10**-9

        #----------get pixel geometry----------
        xAxis = hdf[key]['x_axis']
        yAxis = hdf[key]['y_axis']
        pixelWidth = abs(xAxis[0] - xAxis[1])*siScaling #in meters (assumes that the dimensions of every pixel are the same)
        pixelHeight = abs(yAxis[0] - yAxis[1])*siScaling #in meters
        pixelArea = pixelWidth*pixelHeight #in meters

        return {'units':units, 'pxWidth':pixelWidth, 'pxHeight':pixelHeight, 'pxArea':pixelArea}


    def updateUnits(self):

        #change the apperance of the gui to refelct unit changes, and calculate the pixel equivalent
        #of the selecetd unit

        if not self.lockDict:
            self.lockDict = True
            index = self.fileList.selectedIndexes()[0] #current selected item in gui
            index = index.row()

            #update to units of pixels, so get rid of pixel equivalent string, or restore it to 'No unit data found'
            #and set 'PixelEq' to None, which is the check in runSegmentation() on whether to use the 'haltThresholdSpin'
            #value or the 'pixelEq' value for the halting threshold in the segmentation
            if self.unitsCombo.currentIndex() == 0:
                text = self.parameters[index]['pixelStr']
                if text != 'No unit data found': text = ''
                self.pixelEqLabel.setText(text)
                self.parameters[index]['pixelEq'] = None
                self.lockDict = False
                self.saveParams()
                return

            #find coefficients necessary to convert to meters to match the axis info gathered in getAxisInfo()
            if self.unitsCombo.currentIndex() == 1: #mm^2 in m^2
                conversion = 10**6
            if self.unitsCombo.currentIndex() == 2: #um^2 in m^2
                conversion = 10**12
            if self.unitsCombo.currentIndex() == 3: #nm^2 in m^2
                conversion = 10**18

            #numebr of pixels equal to number of units typed into gui
            pixelsEquivalent = self.haltThresholdSpin.value() / (self.axisInfo[index]['pxArea'] * (conversion))
            text = '= {} pixels'.format('%.2f' % pixelsEquivalent)
            self.parameters[index]['pixelEq'] = int(round(pixelsEquivalent))
            self.parameters[index]['pixelStr'] = text
            self.pixelEqLabel.setText(text)
            self.lockDict = False
            self.saveParams()
        return


    def recalcUnits(self, index):

        #This function behaves like updateUnits(), but doesn't modify the gui or call saveParams(). This is only called
        #when saveAllParams() or saveAllSubParams() is called. Since both of these functions copy the parameters of one image
        #to other images, this function determines how to copy that new threshold value accurately (the pixel equvalent of the
        #halting threshold may be different on two images, but this ensures that the user selected size is always the same in
        #the units selected).

        if self.parameterDict['units'] == 0:
            return '', None

        if self.parameterDict['units'] == 1: #mm^2 in m^2
            conversion = 10**6
        if self.parameterDict['units'] == 2: #um^2 in m^2
            conversion = 10**12
        if self.parameterDict['units'] == 3: #nm^2 in m^2
            conversion = 10**18

        pixelsEquivalent = self.haltThresholdSpin.value() / (self.axisInfo[index]['pxArea'] * (conversion))
        text = '= {} pixels'.format('%.2f' % pixelsEquivalent)
        eq = int(round(pixelsEquivalent))
        return text, eq


    def changeParamsApperance(self):

        #this function modifies the apperance of the parameters frame to reflect the chosen parameters of the highlighted file at index i of the fileList,
        #by loading the parameters from dictionary i in self.parameters

        self.lockDict = True #changing values of the widgets below will result in several calls to saveParams(), which we don't want, because that function
                             #should be reserved for when a USER changes the values, so we lock self.parameterDict from being modified during this

        self.paramsBox.setDisabled(False)

        index = self.fileList.selectedIndexes()[0]
        name = self.fileList.itemFromIndex(index).text()
        index = index.row()
        self.currentFileLabel.setText(name)
        this = self.parameters[index]

        if self.customParams[index] == True:
            self.fileSettingsLabel.setText('Custom settings')
            self.fileSettingsLabel.setPalette(self.blue)
        else:
            self.fileSettingsLabel.setText('Default settings')
            self.fileSettingsLabel.setPalette(self.red)

        #essentailly "loads" the state of all the gui widgets as defined by the dictionary at the current index of self.parameters (which we call 'this')
        if this['smoothingIterations'] != 0: self.smoothCheck.setChecked(True)
        else: self.smoothCheck.setChecked(False)
        self.divideTypeCombo.setCurrentIndex(this['divideType'])
        self.maxPixelDistSpin.setValue(this['maxPixelDist'])
        self.smoothingIterationsSpin.setValue(this['smoothingIterations'])
        self.haltThresholdSpin.setValue(this['haltThreshold'])
        self.bgVarianceSpin.setValue(this['varThreshold'])
        self.bgIntensitySpin.setValue(this['intThreshold'])
        self.unitsCombo.setCurrentIndex(this['units'])
        self.pixelEqLabel.setText(this['pixelStr'])

        if self.axisInfo[index] == None: #image at current index has no unit data
            self.unitsCombo.setCurrentIndex(0)
            self.unitsCombo.setDisabled(True)
            self.lockDict = False
            self.saveParams()
            return

        self.unitsCombo.setDisabled(False)
        self.lockDict = False
        self.saveParams()
        return


    def saveParams(self):

        #this is called every time any paramter is changed, and modifies the dictionary at index i of the self.parameters list
        #(when the file highlighted in the fileList is at index i of the list)
        #The dictionary at the current index of self.parameters is then constantly updated to reflect the users parameter
        #choices for the image at that index
        #Again, self.parameterDict is a temporary dict that will change often, and is just used to gather info from the gui before
        #"saving" it at a specific index of self.paramters

        if not self.lockDict:
            index = self.fileList.selectedIndexes()[0].row()
            self.parameterDict['divideType'] = int(self.divideTypeCombo.currentText())
            self.parameterDict['maxPixelDist'] = self.maxPixelDistSpin.value()
            self.parameterDict['smoothingIterations'] = self.smoothingIterationsSpin.value()
            self.parameterDict['haltThreshold'] = self.haltThresholdSpin.value()
            self.parameterDict['varThreshold'] = self.bgVarianceSpin.value()
            self.parameterDict['intThreshold'] = self.bgIntensitySpin.value()
            self.parameterDict['units'] = self.unitsCombo.currentIndex()
            self.parameterDict['pixelEq'] = self.parameters[index]['pixelEq']
            self.parameterDict['pixelStr'] = self.parameters[index]['pixelStr']
            self.fileSettingsLabel.setText('Custom settings')
            self.fileSettingsLabel.setPalette(self.blue)
            self.parameters[index].update(self.parameterDict)
            self.customParams[index] = True


    def saveAllParams(self):

        #copy the current selected parameters to all other images in the batch. Unit data will be handled specially, so units won't be
        #set for an image with no unit data, and units will be recalculated properly for images with different real pixel sizes.

        for index in range(len(self.parameters)):
            this = self.parameters[index]
            this.update(self.parameterDict)
            if self.axisInfo[index] == None and self.parameterDict['pixelEq'] != None: #meaning image has no unit info, can only have units of pixels
                this['units'] = 0
                this['haltThreshold'] = self.parameterDict['pixelEq'] #can be changed, but this would probably be the user-expected behavior
                this['pixelEq'] = None
                this['pixelStr'] = 'No unit data found'
            else:
                text, eq = self.recalcUnits(index)
                this['pixelStr'] = text
                this['pixelEq'] = eq
            self.customParams[index] = True


    def saveAllSubParams(self):

        #same as above, but for smaller selection

        current = self.fileList.selectedIndexes()[0].row()

        for index in range(len(self.parameters)):
            if index > current:
                this = self.parameters[index]
                this.update(self.parameterDict)
                if self.axisInfo[index] == None and self.parameterDict['pixelEq'] != None:
                    this['units'] = 0
                    this['haltThreshold'] = self.parameterDict['pixelEq']
                    this['pixelEq'] = None
                    this['pixelStr'] = 'No unit data found'
                else:
                    text, eq = self.recalcUnits(index)
                    this['pixelStr'] = text
                    this['pixelEq'] = eq
                self.customParams[index] = True


    def openSegmentDirectory(self):
        if platform.system() == "Windows":
            for path in self.segmentPaths:
                os.startfile(path)
        elif platform.system() == "Linux":
            for path in self.segmentPaths:
                subprocess.Popen(['xdg-open', path])
        else:
            for path in self.segmentPaths:
                os.system(['open "%s"' % path])


    def runSegmentation(self):

        #Runs the segmentation by calling segment_test and passing the neccessary parameters, only
        # after the gui is checked to be in a valid state that won't cause errors

        self.displayPlts = self.plotsCheck.isChecked()
        self.displayLog = self.logCheck.isChecked()
        self.displayRawData = self.rawDataCheck.isChecked()


        if(self.allValid()):
            self.disableAll()
            self.progress.setDisabled(False)
            self.logProgress.setDisabled(False)

            if(self.displayLog):
                #Autmatically switch to the activity log frame if "Display Log" is checked
                self.logView()

            #-------------------- START SEGMENTATION PROCESS -------------------
            t0 = time.time()
            for i in range(len(self.dataPaths)):

                this = self.parameters[i] #current image
                self.divideType = this['divideType']
                self.maxPixelDist = this['maxPixelDist']
                self.smoothingIterations = this['smoothingIterations']
                if this['pixelEq']==None: self.haltThreshold = this['haltThreshold']
                else: self.haltThreshold = this['pixelEq'] #if the user has selected other units, the halt threshold spin box in the gui
                self.maxVar = this['varThreshold']    #is not displaying its value in pixels, so we cannot use it for segmentation,
                self.maxInt = this['intThreshold']    #we need the number in pixels (which is pixelEq in the dict)

                self.currentData = i
                dataPath = self.dataPaths[i]
                imageName = os.path.split(dataPath)[1]

                #gui recieves a list of segmentation results as return, and stores it as an entry in 'segmentData'
                nextData = segmentation.start(dataPath, self.divideType, self.maxPixelDist, self.smoothingIterations, self.displayPlts, self.haltThreshold, len(self.dataPaths))
                self.segmentData.append(nextData)
                nextResults = finalize.finish(self.segmentData[i], self.maxVar, self.maxInt) #find all final segments, find background
                self.results.append(nextResults)
                self.makeSegmentMap(imageName)
                output.toMAPS(nextResults[0], nextResults[1], nextData[3], nextData[1], imageName) #export to MAPS ROI's, pass segments, dimensions, and segmentDir
                gui.advanceProgressBar((100-gui.progress.value())/len(self.dataPaths)) #if len(self.imaegPaths) = 1, this will set the progress bar to 100
                gui.updateLog('Done')

            t2 = '%.2f' % (time.time() - t0)
            self.showMessage('Done', 'All Images finished (took {} seconds).\n Click the results icon in the toolbar to view segmentation maps.'.format(t2), 'message')

            #-------------------- FINISH SEGMENTATION PROCESS -------------------

            #Finishing stuff
            self.progress.setValue(100)
            self.advanceProgressBar(0)
            if self.displayPlts:
                self.resultTabs.setTabEnabled(1, True)
                self.resultTabs.setTabEnabled(2, True)
            if self.displayRawData:
                raw = ''.join(self.rawData)
                self.rawDataDisplay.setPlainText(raw)
                self.resultTabs.setTabEnabled(3, True)

            self.isReset = False
            self.noResults.hide()
            self.openSegmentDir.setDisabled(False)


        return

#---------------------------------------- HELPER FUNCTIONS ----------------------------------------


    def disableAll(self):
        #Disables all main frame widgets during segmentation
        self.segment.setDisabled(True)
        self.reset.setDisabled(True)
        self.openImgButton.setDisabled(True)
        self.openHDF5Button.setDisabled(True)
        self.fileList.setDisabled(True)
        self.smoothCheck.setDisabled(True)
        self.smoothingIterationsSpin.setDisabled(True)
        self.maxPixelDistSpin.setDisabled(True)
        self.bgVarianceSpin.setDisabled(True)
        self.bgIntensitySpin.setDisabled(True)
        self.saveParamsBtn1.setDisabled(True)
        self.saveParamsBtn2.setDisabled(True)
        self.plotsCheck.setDisabled(True)
        self.rawDataCheck.setDisabled(True)
        self.divideTypeCombo.setDisabled(True)
        self.logCheck.setDisabled(True)
        self.toolBar.setDisabled(True)
        self.reset.setDisabled(True)
        self.resultsReset.setDisabled(True)
        self.logReset.setDisabled(True)
        return


    def enableAll(self):

        #Enabled all main frame widgets after segmentation completion
        self.segment.setDisabled(False)
        self.reset.setDisabled(False)
        self.openImgButton.setDisabled(False)
        self.openHDF5Button.setDisabled(False)
        self.fileList.setDisabled(False)
        self.smoothCheck.setDisabled(False)
        self.logCheck.setDisabled(False)
        self.smoothingIterationsSpin.setDisabled(False)
        self.maxPixelDistSpin.setDisabled(False)
        self.bgVarianceSpin.setDisabled(False)
        self.bgIntensitySpin.setDisabled(False)
        self.saveParamsBtn1.setDisabled(False)
        self.saveParamsBtn2.setDisabled(False)
        self.plotsCheck.setDisabled(False)
        self.rawDataCheck.setDisabled(False)
        self.divideTypeCombo.setDisabled(False)
        self.toolBar.setDisabled(False)
        self.reset.setDisabled(False)
        self.resultsReset.setDisabled(False)
        self.logReset.setDisabled(False)
        return


    def allValid(self):

        #Check for any potential errors before segmenting
        if(len(self.dataPaths) == 0):
            self.showMessage('Error', 'No images selected', 'message')
            return  False
        else:
            return True


    def toDefaultParams(self):
        self.parameterDict = {'divideType': 2, 'maxPixelDist': 8, 'units':0, 'pixelEq': None, 'pixelStr':'',
                              'smoothingIterations': 0, 'haltThreshold':100, 'varThreshold':110, 'intThreshold':15}
        return


    def advanceProgressBar(self, amount):
        #changes the grpahic of the progress bar by adding 'amount' percent
        self.progress.setValue(self.progress.value()+amount)
        self.logProgress.setValue(self.progress.value())
        if(self.progress.value() == 100):
            self.enableAll()
        app.processEvents()
        return


    def updateLog(self, string):
        #adds lines to the activity log
        self.log.appendPlainText(string)
        app.processEvents()
        return


    def showMessage(self, title, string, type):

        #Message strings can be passed to this function from any module as to prevent having to import PyQt everywhere to display messages.
        #'warning' and 'question' return an answer to a prompt (either 'Ok'/'Cancel' or 'Yes'/'No') while 'message' simply displays the given string

        if(type == 'warning'):
            answer = qt.QMessageBox.warning(self, title, string, 'Ok', 'Cancel')
            return answer
        if(type == 'question'):
            answer = qt.QMessageBox.warning(self, title, string, 'Yes', 'No')
            return answer
        if(type == 'message'):
            qt.QMessageBox.information(self, title, string, 'Ok')
            return


    def copyLogContents(self):
        #Copy contents of activity log to clipboard
        self.log.copy()
        return


    def openResultsFromFile(self):
        #Open the results of an old segmentation to view in the Results frame
        self.showMessage('Error', 'Not yet implemented', 'message')
        return


    def addPlots(self, figures, plotType):
        #adds a list of scatter figures built in plotframe.py to the results frame
        #plotType: 0 = scatter, 1 = histogram

        i = 1
        j = 0
        #maxColumn is how far over the images will line to the right before starting a "new line" of plots
        maxColumn = 2

        if(plotType == 0):
            frame = self.scatterFrame
            layout = self.scatterLayout
        else:
            frame = self.histogramFrame
            layout = self.histogramLayout

        for fig in figures:
            figure = qt.QFrame(frame)
            figureLayout = qt.QVBoxLayout(figure)
            canvas = FigureCanvas(fig)
            toolbar = NavBar(canvas, figure)
            figureLayout.addWidget(toolbar)
            figureLayout.addWidget(canvas)
            figure.setMinimumSize(400,400)
            figure.setMaximumSize(500,500)
            layout.addWidget(figure, i, j)
            if j < maxColumn:
                j += 1
            else:
                j = 0
                i += 1
        return


    def makeSegmentMap(self, imageName):

        #this function turns the map from output.mapBorders() into a color image

        segmentDir, dimensions = self.segmentData[self.currentData][1], self.segmentData[self.currentData][3] #returned from segmentation.start()
        map = self.results[self.currentData][2] #returned from finalize.finish()
        width = dimensions[0]
        height = dimensions[1]
        ratio = width/height
        imageSize = width*height
        dest = '{}/segmentMap.png'.format(segmentDir)

        #saving to file as RGBA with half opacity in the Alpha channel; borders are red; backgrounds are orange
        mapColor = numpy.zeros(imageSize, dtype=tuple)

        for index in range(len(map)):
            if map[index] == 1: mapColor[index] = (255,0,0,90) #transparent red for border
            elif map[index] == 2: mapColor[index] = (117,205,255,50) #transparent cyan for background
            else: mapColor[index] = (0,0,0,0) #fully transparent for foreground

        borders = Image.new('RGBA', dimensions) #map
        borders.putdata(mapColor)
        image = Image.open(self.imagePaths[self.currentData]) #original image
        map = Image.new('RGB', (width, height))
        map.paste(image)
        map.paste(borders, (0,0), borders) #paste map onto original image, at position (0,0), using mapImage (alpha channel) as mask
        map.save(dest)

        #Creating pixel map of the saved map file and building a QtLabel to hold it
        mapLabel = qt.QLabel('Segmentation Map for {}'.format(imageName), self.segmentsFrame)
        mapLabel.setFont(self.emphasis1)
        pixMap = qt.QPixmap(dest)
        imageHolder = qt.QLabel(self.segmentsFrame)
        imageHolder.setMinimumSize(450, 450/ratio)
        scaledPixMap = pixMap.scaled(imageHolder.size(), QtCore.Qt.KeepAspectRatio)
        imageHolder.setPixmap(scaledPixMap)
        mapFrame = qt.QFrame(self.segmentsFrame)
        mapLayout = qt.QVBoxLayout(mapFrame)

        #Adding components to reuslts window

        mapLayout.addWidget(mapLabel)
        mapLayout.addWidget(imageHolder)
        self.segmentsLayout.addWidget(mapFrame, self.currentData, 0)


    #not currently called from anywhere, add option for this to gui if desired, or delete otherwise
    def returnDiscretized(self, discretizedPath):
        #Sets the "original image" as the newly created discretized image, if checked
        self.dataPaths[self.currentData] = discretizedPath


    def setSegmentPath(self, path):
        self.segmentPaths.append(path)
        return


    def setRawData(self, data):
        self.rawData.append('Image {}: {}\n\n'.format(os.path.split(self.dataPaths[self.currentData])[-1], data))
        return


    def closeApplication(self):
        answer = qt.QMessageBox.question(self, 'Exit', 'Exit the application?', qt.QMessageBox.No | qt.QMessageBox.Yes)
        if answer == qt.QMessageBox.Yes:
            sys.exit()
        else:
            return



#---------------------------------------- POPUP CLASSES ----------------------------------------


class ExchangePopup(qt.QWidget):

    #insatnces of this class will be shown to the user upon selecting to open a HDF5 file
    #provides a list of all exchange groups within the hdf5 in use, and asks which one to look for datasets, calls selectChannels()

    def __init__(self, hdf, name, size):

        qt.QWidget.__init__(self)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self.filename = name
        self.numFiles = size
        self.hdf = hdf
        self.keys = self.hdf.keys()
        self.found = False
        self.label = qt.QLabel('Choose exchange group from {}\nto be used on {} selected files'.format(self.filename, self.numFiles))
        self.okBtn = qt.QPushButton('Ok', self)
        self.cancelBtn = qt.QPushButton('Cancel', self)
        self.combo = qt.QComboBox(self)
        self.combo.activated.connect(self.displayKeyInfo)
        self.setWindowTitle("Choose Exchange")
        self.datasetsLabel = qt.QLabel('Datasets:')
        self.datasetsLabel.setMaximumHeight(15)
        self.attrsField = qt.QLabel('')
        self.attrsField.setWordWrap(True)
        self.okBtn.clicked.connect(self.done)
        self.cancelBtn.clicked.connect(self.cancel)
        emphasis = qt.QFont('Consolas', 10, qt.QFont.Bold)
        self.label.setFont(emphasis)
        self.datasetsLabel.setFont(emphasis)
        self.build()
        return


    def build(self):

        self.layout = qt.QVBoxLayout(self)
        self.attrsFrame = qt.QFrame(self)
        self.attrsLayout = qt.QVBoxLayout(self.attrsFrame)
        self.buttons = qt.QFrame(self)
        self.bottom = qt.QHBoxLayout(self.buttons)
        self.attrsFrame.setFrameStyle(1)
        self.attrsFrame.setLineWidth(0)
        self.attrsFrame.setMidLineWidth(2)
        #self.attrsFrame.setFixedSize(100,100)

        for key in self.keys:
            if key.find('exchange') != -1:
                self.found = True
                self.combo.addItem(key)
        if not self.found:
            str = 'No exchange groups found'
            self.combo.addItem(str)
            self.combo.setDisabled(True)

        self.attrsLayout.addWidget(self.datasetsLabel)
        self.attrsLayout.addWidget(self.attrsField)
        self.bottom.addWidget(self.okBtn)
        self.bottom.addWidget(self.cancelBtn)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo)
        self.layout.addWidget(self.attrsFrame)
        self.layout.addWidget(self.buttons)
        self.move(qt.QApplication.desktop().screen().rect().center()- self.rect().center())
        self.displayKeyInfo()
        return


    def displayKeyInfo(self):
        self.attrsField.clear()
        currentKey = str(self.combo.currentText())
        datasets = self.hdf[currentKey].keys()
        dataString = ''
        for set in datasets:
            dataString += '    {}\n'.format(set)
        self.attrsField.setText(dataString)


    def done(self):
        if self.found:
            gui.selectChannels(str(self.combo.currentText()))
        return


    def cancel(self):
        self.close()
        return


class ChannelsPopup(qt.QWidget):

    #written in refernce to SimpleView2.py - Hong
    #instances of this class will be shown to the user upon selecting an exchange group within an hdf5
    #provides a panel of check boxes corresponding to each channel of data in this hdf5 exchange group, calls setChannels()

    def __init__(self, channels):

        qt.QWidget.__init__(self)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        self.names=list()
        for i in numpy.arange(100):
            self.names.append("")
        self.channels = channels
        self.stack = True
        self.setWindowTitle("Choose Elements")
        self.label = qt.QLabel('Choose channels to use for segmentation', self)
        self.warningLabel = qt.QLabel('', self)
        self.doneBtn = qt.QPushButton('Done', self)
        self.deselectBtn = qt.QPushButton('Deselect All', self)
        self.selectBtn = qt.QPushButton('Select All', self)
        self.stackOptions1 = qt.QRadioButton('Stack channels to one image for segmentation', self)
        self.stackOptions2 = qt.QRadioButton('Segment channels as individual images', self)
        self.stackOptions1.setChecked(True)

        emphasis = qt.QFont('Consolas', 10, qt.QFont.Bold)
        red = qt.QPalette()
        red.setColor(qt.QPalette.Foreground,QtCore.Qt.red)
        self.warningLabel.setFont(emphasis)
        self.warningLabel.setPalette(red)
        self.label.setFont(emphasis)

        self.doneBtn.clicked.connect(self.done)
        self.deselectBtn.clicked.connect(self.deselectAll)
        self.selectBtn.clicked.connect(self.selectAll)
        self.warningLabel.setVisible(False)
        self.build()
        return


    def build(self):

        self.layout = qt.QVBoxLayout()
        self.gridFrame = qt.QFrame(self)
        self.grid = qt.QGridLayout()
        self.hb=qt.QHBoxLayout()

        #make a bunch of check boxes
        pos = []
        for y in numpy.arange(10):
            for x in numpy.arange(10):
                pos.append((x,y))

        self.boxs = []
        i = 0
        for name in self.names:
            self.boxs.append(qt.QCheckBox(name))
            self.boxs[i].setVisible(False)
            self.grid.addWidget(self.boxs[i], pos[i][0], pos[i][1])
            i += 1

        for i in numpy.arange(len(self.channels)):
            self.boxs[i].setText(self.channels[i])
            self.boxs[i].setChecked(False)
            self.boxs[i].setVisible(True)

        self.gridFrame.setLayout(self.grid)
        self.hb.addWidget(self.doneBtn)
        self.hb.addWidget(self.selectBtn)
        self.hb.addWidget(self.deselectBtn)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.gridFrame)
        self.layout.addWidget(self.stackOptions1)
        self.layout.addWidget(self.stackOptions2)
        self.layout.addWidget(self.warningLabel)
        self.layout.addLayout(self.hb)
        self.setLayout(self.layout)
        self.move(qt.QApplication.desktop().screen().rect().center()- self.rect().center())
        return


    def deselectAll(self):
        for i in numpy.arange(len(self.channels)):
            self.boxs[i].setChecked(False)
        return


    def selectAll(self):
        for i in numpy.arange(len(self.channels)):
            self.boxs[i].setChecked(True)
        return


    def done(self):

        selectedIndices = []
        selectedChannels = []

        for i in range(len(self.boxs)):
            if self.boxs[i].isChecked():
                selectedIndices.append(i)
                selectedChannels.append(str(self.boxs[i].text()))

        #check for potential errors before finishing
        if(len(selectedChannels) == 0):
            self.warningLabel.setText('Select at least one channel')
            self.warningLabel.setVisible(True)
            return
        if self.stackOptions1.isChecked():
            if(len(selectedIndices) > 1): self.stack = True
            else:
                self.warningLabel.setText('Select at least two channels to stack')
                self.warningLabel.setVisible(True)
                return
        elif self.stackOptions2.isChecked(): self.stack = False

        gui.setChannels(selectedChannels, selectedIndices, self.stack)
        return


#---------------------------------------- RUN PROGRAM ---------------------------------------

if __name__ == "__main__":

    app = qt.QApplication(sys.argv)
    displaySize = app.desktop().screenGeometry()
    gui = XSDImageSegmentation(displaySize)
    shareGui.setGui(gui)
    sys.exit(app.exec_())

