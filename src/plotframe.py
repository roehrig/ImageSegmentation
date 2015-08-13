__author__ = 'roehrig'


import matplotlib.pyplot as plt
import numpy
import math
import shareGui

def clearAll():
    plt.close('all')

class ScatterPlot ():
    '''
    classdocs
    '''

    def __init__(self, title, data):
        '''
        :param title: The image file that this plot represents
        :param data: The data to be plotted on the Y axis
        :return: Nothing
        '''
        x_values = numpy.arange(data.size)
        y_values = numpy.ravel(data)
        self.area = 20
        self.figure_number = 1


        plt.figure(self.figure_number)
        plt.subplot(111)
        plt.scatter(x_values, y_values, s=self.area)
        plt.title(title)
        return

    def AddPlot(self, title, data):

        x_values = numpy.arange(data.size)
        y_values = numpy.ravel(data)

        self.figure_number = (plt.get_fignums()[-1]) + 1
        plt.figure(self.figure_number)
        plt.subplot(111)
        plt.scatter(x_values, y_values, s=self.area)
        plt.title(title)
        return

    def displayPlots(self):

        figures = plt.get_fignums()
        scatterFignums = []

        for figure in figures:
            if figure % 2 == 1:
                #on every iteration, the scatter plot is created first, and fignums starts at 1, so if
                #the current fig_num % 2 = 1, then that figure is a scatter plot
                scatterFignums.append(figure)

        #scatterFigures is a list of matplotlib figures/their emory locations
        scatterFigures = list(map(plt.figure, scatterFignums))

        #The 0 passed to gui.addPlots() indicates that the list being passed is a list of scatters
        shareGui.getGui().addPlots(scatterFigures, 0)
        return



class HistogramPlot():
    '''
    classdocs
    '''


    def __init__(self, title, data):
        '''

        :param title: The image file that this plot represents
        :param data: The data to used in computing the histogram
        :return:
        '''
        self.figure_number = 2
        plt.figure(self.figure_number)
        plt.subplot(111)
        plt.title(title)
        num_bins = math.ceil(math.log(data.size, 2) + 1)
        arr, bins, patches = plt.hist(numpy.ravel(data), num_bins)
        return


    def AddPlot(self, title, data):

        self.figure_number = (plt.get_fignums()[-1]) + 1
        plt.figure(self.figure_number)
        plt.subplot(111)
        num_bins = math.ceil(math.log(data.size, 2) + 1)
        arr, bins, patches = plt.hist(numpy.ravel(data), num_bins)
        plt.title(title)
        return


    def displayPlots(self):

        figures = plt.get_fignums()
        histogramFignums = []

        for figure in figures:
            if figure % 2 == 0:
                histogramFignums.append(figure)

        histogramFigures = list(map(plt.figure, histogramFignums))

        shareGui.getGui().addPlots(histogramFigures, 1)
        return