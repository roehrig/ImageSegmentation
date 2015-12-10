__author__ = 'hollowed'

import numpy
import os
import pdb
from PIL import Image
import shareGui

#--------------------------------------------------------------------------------------------------------------------------------------

#Runs all finishing processes
def finish(segmentData):

    branches, segmentDir, image, dimensions = segmentData[0], segmentData[1], segmentData[2], segmentData[3]

    finalSegments, finalData = getFinalSegments(branches, segmentDir)
    finalBackground = findBackground(finalData)
    finalMap = mapBorders(segmentDir, dimensions, finalSegments, finalBackground)

    results = [finalSegments, finalBackground, finalMap]
    return results

#--------------------------------------------------------------------------------------------------------------------------------------

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


    print('Algorithm finished with {} final-size segments'.format(len(finalSegments))) #change to gui log

    return finalSegments, finalData

#--------------------------------------------------------------------------------------------------------------------------------------

#flags segments as background or foreground
def findBackground(finalData):

    gui = shareGui.getGui()
    print('Checking for background segments') # change to gui log

    background = numpy.zeros(len(finalData), dtype = int)

    #if the current segment has a variance and mean intensity below both of the threshold values, it is set to '1' in teh background list,
    #indicating that the segment at that index is a background segment
    for n in range(len(finalData)):
        if(numpy.var(finalData[n]) < 110 and numpy.mean(finalData[n]) < 16):
            background[n] = 1

    return background

#--------------------------------------------------------------------------------------------------------------------------------------

#calculates a border map of all final segments
def mapBorders(segmentDir, dimensions, finalSegments, finalBackground):

    width = dimensions[0]
    height = dimensions[1]
    isBackground = False
    imageSize = width*height
    map = numpy.zeros(imageSize, dtype=int)


    for n in range(len(finalSegments)):
        segment = finalSegments[n]
        if(finalBackground[n] == 1): isBackground = True
        else: isBackground = False
        tempMap = numpy.zeros(imageSize, dtype=int)

        #'segment' is one entry in finalSegments, which is a list of lists, each list (segment) containing the pixel indices of that final segment.
        #so, we take each one of those indices, and set it to '1' on the map, meaning that that index in the original image has a pixel in this segment.
        #If that pixel is a background pixel, then we set it to '2'
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
                elif(isBackground):
                    map[pixel] = 2

    #we are now left with a map (2-D array with the original image dimensions) with a '0' for a foreground pixel, a '1' for a border pixel, and a '2' for a background pixel
    return map