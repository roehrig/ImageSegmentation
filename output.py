__author__ = 'hollowed'

import numpy
import os
import pdb
from PIL import Image


def finish(segmentData):

    branches, segmentDir, image, dimensions = segmentData[0], segmentData[1], segmentData[2], segmentData[3]
    finalSegments = getFinalSegments(branches, segmentDir)

    results = [finalSegments]
    return results


def getFinalSegments(branches, segmentDir):

    dirs = next(os.walk(segmentDir))[1]
    allPixelPaths = []
    finalSegments = []

    i = 1
    for dir in dirs:
        if dir.find('pixels_') != -1:
            dir = '/{}'.format(dir)
            pixelPaths = next(os.walk('{}{}'.format(segmentDir, dir)))[2]
            pixelPaths.sort()
            for n in range(len(pixelPaths)):
                pixelPaths[n] = segmentDir + dir + '/' + pixelPaths[n]
            allPixelPaths.extend(pixelPaths)
        i+=1

    for n in range(len(branches)):
        if(branches[n] == 0):
            finalSegments.append((numpy.load(allPixelPaths[n]))['locations'])

    return finalSegments


