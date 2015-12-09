__author__ = 'hollowed'

import sys
import os
import subprocess
import platform
import segment_test
import plotframe
import shareGui
import time
from PIL import Image
from PyQt4 import QtGui as qt, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavBar
import output
import numpy


class XSDImageSegmentation(qt.QMainWindow):

    def __init__(self, displaySize):

        qt.QMainWindow.__init__(self)

        #Creating main app window
        self.WIDTH = 900
        self.HEIGHT = 600
        x = (displaySize.width()/2) - self.WIDTH/2
        y = (displaySize.height()/2) - self.HEIGHT/2
        self.setGeometry(x, y, self.WIDTH, self.HEIGHT)
        self.setWindowTitle("XSD Image Segmentation")
        self.setWindowIcon(qt.QIcon('icon.png'))
        self.type = 0
        #type: 0=single image, 1=multi image
        self.isReset = True
        self.imagePath = None
        self.filename = None
        self.filenames = []
        self.segmentPath = None
        self.divideType = 0
        self.maxPixelDist = 0
        self.smoothingIterations = 0
        self.discretize = False
        self.displayLog = True
        self.displayPlts = False
        self.displayRawData = False
        self.rawData = None

        self.frames()
        self.mainMenu()
        self.toolbar()
        self.buildMainFrame()
        self.buildLogFrame()
        self.buildResultsFrame()
        self.show()

#------------------------------CREATING MAIN LEVEL COMPONENTS------------------------------

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

        #Styling, and addind all fames to StackedWidget for easy switching
        for frame in frames:
            frame.setFrameStyle(qt.QFrame.Panel)
            frame.setFrameShadow(qt.QFrame.Sunken)
            frame.setLineWidth(3)
            self.frameStack.addWidget(frame)

        #Creating fonts
        self.header = qt.QFont('Consolas', 12, qt.QFont.Bold)
        self.emphasis1 = qt.QFont('Consolas', 10, qt.QFont.Bold)
        self.emphasis2 = qt.QFont('Consolas', 10)
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
        self.singleImageTool = qt.QAction(qt.QIcon('singleTool.png'), 'Import a single image for segmentation', self)
        self.multiImageTool = qt.QAction(qt.QIcon('multiTool.png'), 'Import multiple images for segmentation', self)
        self.logTool = qt.QAction(qt.QIcon('logTool.png'), 'View activity log', self)
        self.resultsTool = qt.QAction(qt.QIcon('resultsTool.png'), 'View results panel', self)

        #Assigning actions
        self.singleImageTool.triggered.connect(self.singleImageView)
        self.multiImageTool.triggered.connect(self.multiImageView)
        self.logTool.triggered.connect(self.logView)
        self.resultsTool.triggered.connect(self.resultsView)

        #Populating toolbar
        self.toolBar.addAction(self.singleImageTool)
        self.toolBar.addAction(self.multiImageTool)
        self.toolBar.addAction(self.logTool)
        self.toolBar.addAction(self.resultsTool)
        return

#------------------------------BUILDING ALL PAGES------------------------------

    def buildMainFrame(self):

        #This frame displays the main UI, where images are uploaded, and parameters set

        #Creating frames/layouts to hold components
        centerFrame = qt.QFrame(self.mainFrame)
        filesBox = qt.QGroupBox('Files', centerFrame)
        leftFrame = qt.QFrame(centerFrame)
        paramsBox = qt.QGroupBox('Parameters', leftFrame)
        outputBox = qt.QGroupBox('Output', leftFrame)
        bottomFrame = qt.QFrame(self.mainFrame)

        mainLayout = qt.QVBoxLayout(self.mainFrame) #main frame layout
        centerLayout = qt.QHBoxLayout(centerFrame)
        filesLayout = qt.QVBoxLayout(filesBox)
        leftLayout = qt.QVBoxLayout(leftFrame)
        paramsLayout = qt.QGridLayout(paramsBox)
        outputLayout = qt.QGridLayout(outputBox)
        bottomLayout = qt.QHBoxLayout(bottomFrame)

        #Creating components
        bottomFrame.setMaximumHeight(40)
        self.mainTitle = qt.QLabel('Single Image Segmentation', self.mainFrame)
        self.mainTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.open = qt.QPushButton('Open file...', filesBox)
        self.fileLabel = qt.QLabel('Filename: \n{}'.format(self.filename), filesBox)
        self.fileScroll = qt.QScrollArea(filesBox)
        self.fileList = qt.QListWidget(self.fileScroll)
        self.divideTypePrompt = qt.QLabel('Divide Type:', paramsBox)
        self.divideTypeCombo = qt.QComboBox(paramsBox)
        self.maxPixelDistPrompt = qt.QLabel('Maximum Pixel Distance:', paramsBox)
        self.maxPixelDistSpin = qt.QSpinBox(paramsBox)
        self.discretizeCheck = qt.QCheckBox('Discretize image', paramsBox)
        self.smoothCheck = qt.QCheckBox('Smooth image', paramsBox)
        self.smoothingIterationsPrompt = qt.QLabel('Smoothing Iterations:', paramsBox)
        self.smoothingIterationsSpin = qt.QSpinBox(paramsBox)
        self.minSizePrompt = qt.QLabel('Minimum Segment Size:', paramsBox)
        self.minSizeSpin = qt.QSpinBox(paramsBox)
        self.logCheck = qt.QCheckBox('Display activity log', outputBox)
        self.plotsCheck = qt.QCheckBox('Display Plots', outputBox)
        self.rawDataCheck = qt.QCheckBox('Display raw image data', outputBox)
        self.showMapRadio = qt.QRadioButton('Display segment map', outputBox)
        self.showSegmentsRadio = qt.QRadioButton('Display all segments', outputBox)
        self.segment = qt.QPushButton('Segment', bottomFrame)
        self.segment.setStyleSheet('background-color: lightgreen')
        self.reset = qt.QPushButton('Reset', bottomFrame)
        self.reset.setStyleSheet('background-color: lightblue')
        self.progress = qt.QProgressBar(bottomFrame)

        #Configuring components
        self.open.clicked.connect(self.openImage)
        self.reset.clicked.connect(self.resetAll)
        self.segment.clicked.connect(self.runSegmentation)
        self.smoothCheck.toggled.connect(self.toggleSmoothing)
        self.maxPixelDistSpin.setMinimum(2)
        self.smoothingIterationsSpin.setMinimum(0)
        self.mainTitle.setMaximumHeight(40)
        self.mainTitle.setFont(self.header)
        self.fileLabel.setFont(self.emphasis2)
        self.fileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.fileScroll.setWidgetResizable(True)
        self.fileScroll.setWidget(self.fileList)
        self.fileScroll.hide()
        self.divideTypeCombo.addItems(['0','1','2'])
        self.smoothingIterationsPrompt.setDisabled(True)
        self.smoothingIterationsSpin.setDisabled(True)
        self.minSizeSpin.setValue(1)
        self.minSizeSpin.setMinimum(1)
        self.minSizeSpin.setMaximum(9999)
        self.showMapRadio.setChecked(True)
        self.showSegmentsRadio.setChecked(False)
        self.progress.setValue(0)
        self.progress.setDisabled(True)
        self.logCheck.setChecked(True)
        self.plotsCheck.setChecked(False)
        self.rawDataCheck.setChecked(False)
        self.divideTypeCombo.setCurrentIndex(2)
        self.maxPixelDistSpin.setValue(8)

        #Packing components
        filesLayout.addWidget(self.open)
        filesLayout.addWidget(self.fileLabel)
        filesLayout.addWidget(self.fileScroll)
        paramsLayout.addWidget(self.divideTypePrompt, 0, 0)
        paramsLayout.addWidget(self.divideTypeCombo, 0, 1)
        paramsLayout.addWidget(self.maxPixelDistPrompt, 1, 0)
        paramsLayout.addWidget(self.maxPixelDistSpin, 1, 1)
        paramsLayout.addWidget(self.smoothCheck, 3, 0)
        paramsLayout.addWidget(self.smoothingIterationsPrompt, 2, 1)
        paramsLayout.addWidget(self.smoothingIterationsSpin, 3, 1)
        paramsLayout.addWidget(self.discretizeCheck, 2, 0)
        paramsLayout.addWidget(self.minSizePrompt, 5, 0)
        paramsLayout.addWidget(self.minSizeSpin, 5, 1)
        outputLayout.addWidget(self.logCheck, 1, 0)
        outputLayout.addWidget(self.plotsCheck, 2,0)
        outputLayout.addWidget(self.rawDataCheck, 3, 0)
        outputLayout.addWidget(self.showMapRadio, 4, 0)
        outputLayout.addWidget(self.showSegmentsRadio, 5, 0)
        leftLayout.addWidget(paramsBox)
        leftLayout.addWidget(outputBox)
        centerLayout.addWidget(filesBox)
        centerLayout.addWidget(leftFrame)
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
        self.resultsTitle.setFont(self.header)
        self.resultsTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.noResults.setAlignment(QtCore.Qt.AlignCenter)
        self.resultsReset.clicked.connect(self.resetAll)
        self.resultsReset.setStyleSheet('background-color: lightblue')
        self.resultsReset.setMaximumWidth(80)
        self.openSegmentDir.clicked.connect(self.openSegmentDirectory)
        self.openSegmentDir.setDisabled(True)
        self.openSegmentDir.setMaximumWidth(160)
        self.resultTabs.setTabEnabled(1, False)
        self.resultTabs.setTabEnabled(2, False)
        self.resultTabs.setTabEnabled(3, False)
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

#------------------------------TOOLBAR BUTTON FUNCTIONS------------------------------

    def multiImageView(self):

        #Adjusts the neccessary widgets when the main frame is switched to multi-image
        self.frameStack.setCurrentWidget(self.mainFrame)
        self.type = 1
        self.mainTitle.setText('Multiple Image Segmentation')
        self.fileLabel.setText('Filenames:')
        self.fileLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.open.setText('Open files...')
        self.fileScroll.show()
        if(len(self.filenames) == 0):
            self.fileLabel.setFont(self.emphasis2)
        else:
            self.fileLabel.setFont(self.emphasis1)
        return


    def singleImageView(self):

        #Adjusts the neccessary widgets when the main frame is switched to single-image
        self.frameStack.setCurrentWidget(self.mainFrame)
        self.type = 0
        self.mainTitle.setText('Single Image Segmentation')
        self.fileLabel.setText('Filename: \n{}'.format(self.filename))
        self.fileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.open.setText('Open file...')
        self.fileScroll.hide()
        if(self.filename == None):
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


#------------------------------ WIDGET ACTION FUNCTIONS ------------------------------

    def resetAll(self):

        #Resets everything in the gui to its initial state
        self.enableAll()

        self.imagePath = None
        self.divideType = 0
        self.maxPixelDist = 0
        self.filename = None
        self.filenames = []
        self.smoothingIterations = 0
        self.discretize = False
        self.displayLog = True
        self.displayPlts = False
        self.displayRawData = False

        self.divideTypeCombo.setCurrentIndex(2)
        self.maxPixelDistSpin.setValue(8)
        self.minSizeSpin.setValue(1)
        self.logCheck.setChecked(True)
        self.plotsCheck.setChecked(False)
        self.rawDataCheck.setChecked(False)
        self.smoothCheck.setChecked(False)
        self.discretizeCheck.setChecked(False)
        self.smoothingIterationsSpin.setValue(0)
        self.smoothingIterationsPrompt.setDisabled(True)
        self.smoothingIterationsSpin.setDisabled(True)
        self.showMapRadio.setChecked(True)
        self.showSegmentsRadio.setChecked(False)
        self.progress.setValue(0)
        self.progress.setDisabled(True)
        self.logProgress.setValue(0)
        self.logProgress.setDisabled(True)
        self.log.clear()
        self.fileList.clear()
        self.fileLabel.setFont(self.emphasis2)
        self.openSegmentDir.setDisabled(True)

        if(self.type == 0):
            self.fileLabel.setText('Filename: \n{}'.format(self.filename))
        else:
            self.fileLabel.setText('Filenames:')

        #This loop clears all images and plots from the results tabs
        layouts = [self.scatterLayout, self.histogramLayout, self.segmentsLayout]
        for layout in layouts:
            for i in reversed(range(layout.count())):
                #The reversed range is necessary to prevent error
                layout.itemAt(i).widget().setParent(None)

        #calls the clarAll() method in plotframe.py so that every plot is erased
        plotframe.clearAll()
        self.segmentsLayout.addWidget(self.noResults, 0, 0)
        self.scatterLayout.addWidget(self.scatterAxesLabels, 0, 0)
        self.histogramLayout.addWidget(self.histogramAxesLabels, 0, 0)
        self.noResults.show()
        self.resultTabs.setTabEnabled(1, False)
        self.resultTabs.setTabEnabled(2, False)
        self.resultTabs.setTabEnabled(3, False)

        self.isReset = True
        return


    def toggleSmoothing(self):

        if(self.smoothCheck.isChecked()):
            self.smoothingIterationsSpin.setDisabled(False)
            self.smoothingIterationsPrompt.setDisabled(False)
        else:
            self.smoothingIterationsSpin.setDisabled(True)
            self.smoothingIterationsPrompt.setDisabled(True)
        return


    def openImage(self):

        if(self.isReset == False):
            answer = self.showMessage('Error', 'Window must be reset before segmenting again.\nReset Now?', 'question')
            if answer == 0:
                self.resetAll()
                time.sleep(1)
                self.openImage()
            return

        #open the image from file to be segmented
        if(self.type == 0):
            temp = str(qt.QFileDialog.getOpenFileName(self, 'Open Image'))

            if(temp != ''):
                #This statement prevents a bug where the user hits 'cancel' on the file open window
                #after already having opened one previously
                self.imagePath = temp
                self.filename = os.path.split(self.imagePath)[-1]
                self.fileLabel.setText('Filename: \n{}'.format(self.filename))
                self.fileLabel.setFont(self.emphasis1)
        else:
            temp = qt.QFileDialog.getOpenFileNames(self, 'Open Images')

            if(temp != ''):
                imagePaths= temp

            for imagePath in imagePaths:
                self.filenames.append(os.path.split(str(imagePath))[-1])
            for file in self.filenames:
                self.fileList.addItem(file)

            self.imagePath = imagePaths[0]
            self.fileLabel.setFont(self.emphasis1)
        return


    def blowupImage(self, cut):
        def blowUp():
            image = Image.open(cut)
            image.show()
        return blowUp


    def openSegmentDirectory(self):
        if platform.system() == "Windows":
            os.startfile(self.segmentPath)
        elif platform.system() == "Linux":
            subprocess.Popen(['xdg-open', self.segmentPath])
        else:
            #OSX
            os.system(['open "%s"' % self.segmentPath])


    def runSegmentation(self):
        #Runs the segmentation by calling segment_test and passing the neccessary parameters, only
        # after the gui is checked to be in a valid state that won't cause errors
        self.discretize = self.discretizeCheck.isChecked()
        self.divideType = int(self.divideTypeCombo.currentText())
        self.maxPixelDist = self.maxPixelDistSpin.value()
        self.smoothingIterations = self.smoothingIterationsSpin.value()
        self.displayPlts = self.plotsCheck.isChecked()
        self.displayLog = self.logCheck.isChecked()
        self.displayRawData = self.rawDataCheck.isChecked()
        self.minSize = self.minSizeSpin.value()
        self.showSegments = self.showSegmentsRadio.isChecked()
        self.showMap = self.showMapRadio.isChecked()


        if(self.allValid()):
            self.disableAll()
            self.progress.setDisabled(False)
            self.logProgress.setDisabled(False)

            if(self.displayLog):
                #Autmatically switch to the activity log frame if "Display Log" is checked
                self.logView()

            if(self.type == 0):
                #gui recieves a list of segmentation results as return, and stores it as 'segmentData'. Only the first entry in this list is used
                #here in the gui, and the rest (the entire list) is then passed to output.finish() below
                self.segmentData = segment_test.start(self.imagePath, self.divideType, self.maxPixelDist, self.discretize, self.smoothingIterations, self.displayPlts, self.minSize)
                self.branches = self.segmentData[0]

                if self.segmentPath == None:
                    #In case something (probably discretizing) goes wrong
                    self.resetAll()
                    self.frameStack.setCurrentWidget(self.mainFrame)
                    return

            if(self.type == 1):
                self.showMessage('Error', 'Not yet implemented', 'message')
                self.resetAll()
                self.enableAll()
                self.frameStack.setCurrentWidget(self.mainFrame)
                return

            if self.displayPlts:
                self.resultTabs.setTabEnabled(1, True)
                self.resultTabs.setTabEnabled(2, True)
            else:
                self.resultTabs.setTabEnabled(1, False)
                self.resultTabs.setTabEnabled(2, False)


            if self.displayRawData:
                self.rawDataDisplay.setPlainText(str(self.rawData))
                self.resultTabs.setTabEnabled(3, True)
            else:
                self.resultTabs.setTabEnabled(3, False)

            #Nothing went wrong, display results, and run output functions
            self.isReset = False
            self.noResults.hide()
            self.openSegmentDir.setDisabled(False)
            self.results = output.finish(self.segmentData)

            if(self.showSegments):
                self.addImageStructure()
            if(self.showMap):
                self.makeSegmentMap()

        return

#------------------------------ HELPER FUNCTIONS ------------------------------

    def disableAll(self):

        #Disables all main frame widgets during segmentation
        self.segment.setDisabled(True)
        self.reset.setDisabled(True)
        self.open.setDisabled(True)
        self.fileList.setDisabled(True)
        self.discretizeCheck.setDisabled(True)
        self.smoothCheck.setDisabled(True)
        self.smoothingIterationsSpin.setDisabled(True)
        self.maxPixelDistSpin.setDisabled(True)
        self.plotsCheck.setDisabled(True)
        self.rawDataCheck.setDisabled(True)
        self.divideTypeCombo.setDisabled(True)
        self.logCheck.setDisabled(True)
        self.showSegmentsRadio.setDisabled(True)
        self.showMapRadio.setDisabled(True)
        self.toolBar.setDisabled(True)
        self.reset.setDisabled(True)
        self.resultsReset.setDisabled(True)
        self.logReset.setDisabled(True)
        return


    def enableAll(self):

        #Enabled all main frame widgets after segmentation completion
        self.segment.setDisabled(False)
        self.reset.setDisabled(False)
        self.open.setDisabled(False)
        self.fileList.setDisabled(False)
        self.discretizeCheck.setDisabled(False)
        self.smoothCheck.setDisabled(False)
        self.logCheck.setDisabled(False)
        self.smoothingIterationsSpin.setDisabled(False)
        self.maxPixelDistSpin.setDisabled(False)
        self.plotsCheck.setDisabled(False)
        self.rawDataCheck.setDisabled(False)
        self.divideTypeCombo.setDisabled(False)
        self.showSegmentsRadio.setDisabled(False)
        self.showMapRadio.setDisabled(False)
        self.toolBar.setDisabled(False)
        self.reset.setDisabled(False)
        self.resultsReset.setDisabled(False)
        self.logReset.setDisabled(False)
        return


    def allValid(self):

        #Check for any potential errors before segmenting
        if(self.imagePath == None):
            self.showMessage('Error', 'No image selected', 'message')
            return  False
        else:
            return True


    def advanceProgressBar(self, amount):
        #changes the grpahic of the progress bar
        self.progress.setValue(amount)
        self.logProgress.setValue(amount)
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


    def addImageStructure(self):
        #Creates and organizes a structure (grid) for all segmented images written in imagedata.py to be added to the results
        #page under the 'Segmentation' tab
        cuts = []
        c = 1   #cut number
        self.s = 0   #segment number
        i = 1   #row
        j = 0   #column
        originalLabel = qt.QLabel('Original Image', self.segmentsFrame)
        originalLabel.setFont(self.emphasis1)
        self.segmentsLayout.addWidget(originalLabel, 0, 0)
        #self.imagePath is the path to the original image
        cuts.append(self.imagePath)
        self.addImages(cuts, i, j, '>> Open image file')
        #'dirs' gets the list of directories created during segmentation (to get all "cut_n"'s)
        dirs = next(os.walk(self.segmentPath))[1]
        dirs.sort()

        for dir in dirs:
            i += 1
            j = 0
            dir = '/{}'.format(dir)
            if dir.find('/cut_') != -1:
                #'cuts' gets list of all images in the current "cut_n" directory
                cuts = next(os.walk('{}{}'.format(self.segmentPath, dir)))[2]
                cuts.sort()
                for n in range(len(cuts)):
                    cuts[n] = self.segmentPath + ('/cut_%d/' % c) + cuts[n]
                cutLabel = qt.QLabel('\nCut %d' % c, self.segmentsFrame)
                cutLabel.setFont(self.emphasis1)
                self.segmentsLayout.addWidget(cutLabel, i, 0)
                c += 1
                i += 1
                i = self.addImages(cuts, i, j, '>>')

        return


    def addImages(self, cuts, i, j, buttonText):
        #helper function for addImageStructure() above, inserts the images one by one

        width = 150
        height = 150
        #maxColumn is how far over the images will line to the right before starting a "new line" of images
        maxColumn = 3

        for cut in cuts:
            imageFrame = qt.QFrame(self.segmentsFrame)
            imageLayout = qt.QVBoxLayout(imageFrame)
            blowupButton = qt.QPushButton(buttonText, imageFrame)
            blowupButton.setMaximumWidth(width)
            blowupButton.clicked.connect(self.blowupImage(cut))
            blowupButton.setToolTip('Open image file')
            pixMap = qt.QPixmap(cut)
            imageHolder = qt.QLabel(imageFrame)
            imageHolder.setMinimumSize(width, height)
            if(self.s > 0 and self.branches[self.s-1] == 0):
                imageHolder.setStyleSheet('border: 5px solid red;')
            self.s += 1
            scaledMap = pixMap.scaled(imageHolder.size(), QtCore.Qt.KeepAspectRatio)
            imageHolder.setPixmap(scaledMap)
            imageLayout.addWidget(imageHolder)
            imageLayout.addWidget(blowupButton)
            self.segmentsLayout.addWidget(imageFrame, i, j)
            if j < maxColumn:
                j += 1
            else:
                j = 0
                i += 1
        return(i)

    #this function creates a "map" of the final segents on one image
    def makeSegmentMap(self):

        segmentDir, dimensions = self.segmentData[1], self.segmentData[3]
        finalSegments = self.results[0]

        width = dimensions[0]
        height = dimensions[1]
        ratio = width/height
        imageSize = width*height
        map = numpy.zeros(imageSize, dtype=int)
        dest = '{}/segmentMap.png'.format(segmentDir)

        for segment in finalSegments:
            tempMap = numpy.zeros(imageSize, dtype=int)

            #'segment' is one entry in finalSegments, which is a list of lists, each containing the pixel indices of each final segment.
            #so, we take each one of those indices, and set it to '1' on the map.
            for index in segment:
                tempMap[index] = 1

            for pixel in range(len(tempMap)):

                #the neighbor count of each pixel is found. If they have a neighbor on everyside, they are interior to a segment, as is
                #omitted from the map image. If fiding the neighbors returns an IndexError, then this is a border pixel, and can be part of the map
                if tempMap[pixel] == 1:
                    try: neighborCount = tempMap[pixel+1] + tempMap[pixel-1] + tempMap[pixel-width] + tempMap[pixel+width]
                    except IndexError: neighborCount = 0

                    #all pixels that do not have a neighbor on every side must be a boundary pixel of this segment, so it is drawn into the map
                    if(neighborCount != 4):
                        map[pixel] = 1

        #saving to file as RGBA with half opacity in the Alpha channel; borders are red
        picData = numpy.zeros(imageSize, dtype=tuple)
        for index in range(len(map)):
            if map[index] == 1: picData[index] = (255,0,0,175)
            else: picData[index] = (0,0,0,0)
        c = Image.new('RGBA', dimensions)
        c.putdata(picData)
        c.save(dest)

        #Creating pixel map of the saved map file and building a QtLabel to hold it
        mapLabel = qt.QLabel('Segmentation Map', self.segmentsFrame)
        mapLabel.setFont(self.emphasis1)
        pixMap = qt.QPixmap(self.imagePath) #original image
        pixMapOverlay = qt.QPixmap(dest) #segmentation map
        imageHolder = qt.QLabel(self.segmentsFrame)
        imageHolder.setMinimumSize(450, 450/ratio)
        mapHolder = qt.QLabel(self.segmentsFrame)
        mapHolder.setMinimumSize(450, 450/ratio)
        scaledPixMap = pixMap.scaled(imageHolder.size(), QtCore.Qt.KeepAspectRatio)
        scaledPixMapOverlay = pixMapOverlay.scaled(mapHolder.size(), QtCore.Qt.KeepAspectRatio)
        mapHolder.setPixmap(scaledPixMapOverlay)
        imageHolder.setPixmap(scaledPixMap)

        #Adding components to reuslts window
        self.segmentsLayout.addWidget(mapLabel, 0, 0)
        self.segmentsLayout.addWidget(imageHolder, 1, 0)
        self.segmentsLayout.addWidget(mapHolder, 1, 0)


    def returnDiscretized(self, discretizedPath):
        #Sets the "original image" as the newly created discretized image, if checked
        self.imagePath = discretizedPath


    def setSegmentPath(self, path):
        self.segmentPath = path
        return


    def setRawData(self, data):
        self.rawData = data
        return


    def closeApplication(self):
        answer = qt.QMessageBox.question(self, 'Exit', 'Exit the application?', qt.QMessageBox.No | qt.QMessageBox.Yes)
        if answer == qt.QMessageBox.Yes:
            sys.exit()
        else:
            return

#------------------------------ RUN PROGRAM ------------------------------

if __name__ == "__main__":

    app = qt.QApplication(sys.argv)
    displaySize = app.desktop().screenGeometry()
    gui = XSDImageSegmentation(displaySize)
    shareGui.setGui(gui)
    sys.exit(app.exec_())

