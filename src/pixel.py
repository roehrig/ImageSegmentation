__author__ = 'roehrig'

import math
import numpy

class Pixel():

    def __init__(self, value=0, x=0, y=0):

        self.value = value
        self.x = x
        self.y = y

        self.location = math.sqrt(pow(x, 2) + pow(y, 2))

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
                stride = (width * i) + j
                self.pixelList.append(Pixel(values[stride], j, i))

        self.arraySize = len(self.pixelList)

        return

    def GetPixelArray(self):
        return self.pixelList

    def GetPixelArraySize(self):
        return self.arraySize

    def CreateLocationArray(self):

        size = len(self.pixelList)
        locations = numpy.zeros(size)

        for i in range(size):
            locations[i] = self.pixelList[i].GetLocation()

        return locations

