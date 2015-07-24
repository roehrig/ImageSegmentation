__author__ = 'roehrig'

import math
import numpy

class Pixel():

    def __init__(self, value=0, x=0, y=0):

        self.value = value #intensity value
        self.x = x
        self.y = y

        self.location = math.sqrt(pow(x, 2) + pow(y, 2)) #location from top left corner

        return

    def GetValue(self):
        return self.value

    def GetLocation(self):
        return self.location

    def GetColumnNumber(self):
        return self.x

    def GetRowNumber(self):
        return self.y

class PixelArray():

    def __init__(self, width, height, values):

        self.pixelList = []

        for i in range(height):
            for j in range(width):
                stride = (width * i) + j                           # numbers all pixels from top left to bottom right
                self.pixelList.append(Pixel(values[stride], j, i)) #passes the value at the 'stride'th position in the 'values' array,
                                                                   # passed by ImageFileData.ReadImage at creation of data.pixels, to Pixel()

                #end up with a list of all pixels, each holding info on their intensity value, grid position, and relative location

        self.arraySize = len(self.pixelList)

        return

    def GetPixelArray(self):
        return self.pixelList

    def GetPixelArraySize(self):
        return self.arraySize

    def CreateLocationArray(self):

        size = len(self.pixelList)
        locations = numpy.zeros(size) #returns an array of the specified shape, filled with zeroes

        for i in range(size):
            locations[i] = self.pixelList[i].GetLocation()

        return locations #returns array with an item for each pixel, the value of each being its distance from the top left corner of the image

