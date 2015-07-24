__author__ = 'roehrig'

from PIL import Image
from PIL import ImageFilter
from pixel import *

class ImageData():

    def __init__(self, fileName=None, data=None, width=0, height=0):

        self.file = fileName
        self.dataImage = None
        self.data = data
        self.width = width
        self.height = height
        self.pixels = None
        self.imageMode = None
        self.size = width * height

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

    def GetBands(self):
        return self.bands   #gets the number of bands or channels composing the image

    def GetFileFormat(self):
        return self.fileFormat


class ImageArrayData(ImageData):

    def __init__(self, fileName=None, data=None, width=0, height=0):

        ImageData.__init__(fileName, data, width, height)

        return

    def SetImageMode(self, mode):

        self.imageMode = mode
        return

class ImageFileData(ImageData):

    def __init__(self, fileName=None, data=None, width=0, height=0):

        ImageData.__init__(self, fileName, data, width, height)

        return

    def ReadImage(self):

        self.dataImage = Image.open(self.file, mode='r')
        self.width = self.dataImage.size[0]
        self.height = self.dataImage.size[1]
        self.data = self.dataImage.getdata()   #Intensity values per pixel; will return a tuple if image is multi-layer
        self.size = len(self.data)
        self.fileFormat = self.dataImage.format
        self.imageMode = self.dataImage.mode
        self.bands = self.dataImage.getbands()


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

        sorted_values = sorted(self.data)
        pixel_values = self.pixels.GetPixelArray()

        if self.size % 2 == 1:
            median = sorted_values[((self.size + 1) / 2) - 1]
        else:
            median = float((sorted_values[(self.size + 1) / 2] + sorted_values[(self.size - 1) / 2]) / 2.0)

        for i in range (self.height):
            for j in range(self.width):
                stride = (self.width * i) + j
                if self.data[stride] >= median:
                    self.data[stride] = maxVal
                    pixel_values[stride] = maxVal
                else:
                    self.data[stride] = minVal
                    pixel_values[stride] = minVal

        return

    def SmoothImage(self, iterations):

        for i in range(iterations):
            newImage = self.dataImage.filter(ImageFilter.SMOOTH_MORE)

        self.dataImage = newImage
        self.data = self.dataImage.getdata()
        self.pixels = PixelArray(self.width, self.height, self.data)

        return
