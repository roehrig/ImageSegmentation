__author__ = 'roehrig'

from PIL import Image
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
        self.data = self.dataImage.getdata()
        self.size = len(self.data)
        self.fileFormat = self.dataImage.format
        self.imageMode = self.dataImage.mode

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