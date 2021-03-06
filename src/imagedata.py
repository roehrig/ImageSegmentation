__author__ = 'roehrig'

from PIL import Image
from PIL import ImageFilter
from pixel import *
import shareGui
import os
import numpy
import h5py

#--------------------------------------------------------------------------------------------------------------------------------------


class ImageData():

    def __init__(self, fileName=None, segmentDir = None, data=None, width=0, height=0):

        self.file = fileName
        self.segmentDir = segmentDir
        self.dataImage = None
        self.data = data
        self.width = width
        self.height = height
        self.pixels = None
        self.imageMode = None
        self.size = width * height
        self.fileFormat = None
        self.channels = None

        return

    def GetImageData(self):
        return self.data

    def GetImageDimensions(self):
        return (self.width, self.height)

    def GetImageSize(self):
        return self.size

    def GetPixelsArray(self):
        return self.pixels.GetPixelArray()

    def GetPixels(self):
        return self.pixels

    def GetImageMode(self):
        return self.imageMode

    def GetChannels(self):

        return self.channels

    def GetFileFormat(self):
        return self.fileFormat


#--------------------------------------------------------------------------------------------------------------------------------------


class ImageArrayData(ImageData):

    #using a data array for segmentation (HDF5)

    def __init__(self, fileName=None, segmentDir = None, data=None, width=0, height=0):

        ImageData.__init__(self, fileName, segmentDir, data, width, height)
        return

    def ReadImage(self):

        self.imagePath, self.imageFile = os.path.split(self.file)
        self.dataImage = numpy.load(self.file)
        self.data = self.dataImage['data']   #Intensity values per pixel; will return a tuple if image is multi-layer
        self.width = self.dataImage['dimensions'][0]
        self.height = self.dataImage['dimensions'][1]
        self.size = len(self.data)
        self.channels = self.dataImage['channels']
        self.fileFormat = 'npz'
        self.imageMode = 'N/A'
        self.iteration = 1

        # Create a list of Pixel objects (see pixel.py)
        self.pixels = PixelArray(self.width, self.height, self.data)

        return

    def WriteImage(self, fileName=None):
        self.dataImage.save(fileName, self.fileFormat)
        return

    def WriteNewImage(self, data, fileName):
        dataImage = numpy.save('{}.npy'.format(fileName), data)
        return


#---------------------------------------------------------------------------------------------------------------------------------------


class ImageFileData(ImageData):

    #using an image file for segmentation (TIF, PNG, etc.)

    def __init__(self, fileName=None, segmentDir = None, data=None, width=0, height=0):

        ImageData.__init__(self, fileName, segmentDir, data, width, height)
        return

    def ReadImage(self):

        self.imagePath, self.imageFile = os.path.split(self.file)
        self.dataImage = Image.open(self.file, mode='r')
        self.width = self.dataImage.size[0]
        self.height = self.dataImage.size[1]
        self.data = self.dataImage.getdata()   #Intensity values per pixel; will return a tuple if image is multi-layer
        self.size = len(self.data)
        self.channels = self.dataImage.getbands()
        self.fileFormat = self.dataImage.format
        self.imageMode = self.dataImage.mode
        self.iteration = 1

        # Create a list of Pixel objects (see pixel.py)
        self.pixels = PixelArray(self.width, self.height, self.data)

        return

    def WriteImage(self, fileName=None):

        self.dataImage.save(fileName, self.fileFormat)
        return

    def WriteNewImage(self, data, fileName):
        dataImage = Image.new(self.imageMode, self.dataImage.size)
        dataImage.putdata(data)
        dataImage.save(fileName, self.fileFormat)
        return

    def DiscretizeImage(self, maxVal=1, minVal=0):

        '''
        This function takes an image and finds the median pixel value, then compares
        the value of each pixel to the median.  If a pixel is greater than or equal
        to the median, the pixel value is set to a user supplied value.  If the pixel
        value is less than the median, the pixel value is set to another user
        supplied value.

        :param maxVal: the high value use for pixels
        :param minVal: the low value to use for pixels
        :return:
        '''
        gui = shareGui.getGui()

        if len(self.channels) > 1:
            answer = gui.showMessage('Error', 'Discretization can only be preformed on grayscale images.\nThe current image has the following bands: '
                                     '\n{}\nPress \'Ok\' to continue segmentation without discretizing, or \'Cancel\' to abort.'.format(self.channels), 'warning')
            if answer == 0:
                pass
            if answer == 1:
                return False

        shareGui.getGui().updateLog('Discretizing image')

        temp = numpy.zeros(self.size)

        for i in range(self.size):
            temp[i] = self.data[i]

        sorted_values = sorted(self.data)
        pixel_values = self.pixels.GetPixelArray()

        if self.size % 2 == 1:
            median = sorted_values[((self.size + 1) / 2) - 1]
        else:
            median = float((sorted_values[(self.size + 1) / 2] + sorted_values[(self.size - 1) / 2]) / 2.0)

        for i in range (self.height):
            for j in range(self.width):
                stride = (self.width * i) + j
                if temp[stride] >= median:
                    temp[stride] = maxVal
                    pixel_values[stride] = maxVal
                else:
                    temp[stride] = minVal
                    pixel_values[stride] = minVal

        newFile= '{}{}_discretized.{}'.format(self.segmentDir, self.imageFile.split('.')[0], self.fileFormat)
        self.WriteNewImage(temp, newFile)
        gui.returnDiscretized(newFile)
        self.__init__(newFile)
        self.ReadImage()
        return True


    def SmoothImage(self, iterations):

        for i in range(iterations):
            shareGui.getGui().updateLog('Smoothing iteration #%d' % self.iteration)
            newImage = self.dataImage.filter(ImageFilter.SMOOTH_MORE)
            self.iteration += 1

        self.dataImage = newImage
        self.data = self.dataImage.getdata()
        self.pixels = PixelArray(self.width, self.height, self.data)

        return
