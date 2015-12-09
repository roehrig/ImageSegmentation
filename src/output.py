__author__ = 'hollowed'

import numpy
import os
import pdb
from PIL import Image
import shareGui


#Runs all finishing processes
def finish(segmentData):

    branches, segmentDir, image, dimensions = segmentData[0], segmentData[1], segmentData[2], segmentData[3]
    finalSegments, finalData = getFinalSegments(branches, segmentDir)
    backgroundSegments = determineBg(finalData)

    results = [finalSegments, backgroundSegments]
    return results


#Goes through all pixel data from the segmentation and specifies which files correspond to final-size segments
def getFinalSegments(branches, segmentDir):

    gui = shareGui.getGui()

    dirs = next(os.walk(segmentDir))[1]
    allPixelPaths = []
    finalSegments = []
    finalData = []

    #Finding all pixel files
    for dir in dirs:
        if dir.find('pixels_') != -1:
            dir = '/{}'.format(dir)
            pixelPaths = next(os.walk('{}{}'.format(segmentDir, dir)))[2]
            pixelPaths.sort()
            for n in range(len(pixelPaths)):
                pixelPaths[n] = segmentDir + dir + '/' + pixelPaths[n]
            allPixelPaths.extend(pixelPaths)

    #checking to see which pixel files correspond to final-size segments
    for n in range(len(branches)):
        if(branches[n] == 0):
            finalSegments.append((numpy.load(allPixelPaths[n]))['locations'])
            finalData.append((numpy.load(allPixelPaths[n]))['pixels'])


    print('Algorithm finished with {} final-size segments'.format(len(finalSegments)))

    return finalSegments, finalData


#flags segments as background or foreground
def determineBg(finalData):

    gui = shareGui.getGui()
    print('Checking for background segments')

    background = numpy.zeros(len(finalData), dtype = int)

    #if the current segment has a variance and mean intensity below both of the threshold values, it is set to '1' in teh background list,
    #indicating that the segment at that index is a background segment
    for n in range(len(finalData)):
        if(numpy.var(finalData[n]) < 110 and numpy.mean(finalData[n]) < 16):
            background[n] = 1

    return background

