__author__ = 'roehrig'


import matplotlib.pyplot as plt
import numpy
import math
import shareGui

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

        self.figure_number = self.figure_number + 1
        plt.figure(self.figure_number)
        plt.subplot(111)
        plt.scatter(x_values, y_values, s=self.area)

        return

    def displayPlots(self):
        figures = list(map(plt.figure, plt.get_fignums()))
        shareGui.getGui().addPlots(figures, 0)
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

        num_bins = math.ceil(math.log(data.size, 2) + 1)

        arr, bins, patches = plt.hist(numpy.ravel(data), num_bins)
        print arr
        print bins
        print patches

        plt.show()

        return
