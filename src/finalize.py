__author__ = 'hollowed'

import numpy
import os
import shareGui
import shutil

#--------------------------------------------------------------------------------------------------------------------------------------

#Runs all finishing processes
def finish(segmentData, maxVar, maxInt):

    branches, segmentDir, image, dimensions = segmentData[0], segmentData[1], segmentData[2], segmentData[3]
    gui = shareGui.getGui()

    finalSegments, finalData, finalPaths = getFinalSegments(branches, segmentDir)
    finalBackground = findBackground(finalData, maxVar, maxInt)
    finalMap = mapBorders(segmentDir, dimensions, finalSegments, finalBackground)
    cleanup(segmentDir, finalPaths)

    results = [finalSegments, finalBackground, finalMap]
    return results

#--------------------------------------------------------------------------------------------------------------------------------------

#Goes through all pixel data from the segmentation and specifies which files correspond to final-size segments
def getFinalSegments(branches, segmentDir):

    gui = shareGui.getGui()

    dirs = next(os.walk(segmentDir))[1]
    allPixelPaths = []
    finalSegments = [] #each entry is a numpy array (segment) containing pixel indices
    finalData = [] #each entry is a numpy array (segment) containing pixel values
    finalPaths = [] #each entry is a path string to the pixel data of a final segment

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
            finalPaths.append(allPixelPaths[n])
            finalSegments.append((numpy.load(allPixelPaths[n]))['locations'])
            finalData.append((numpy.load(allPixelPaths[n]))['pixels'])

    finalSegments = numpy.array(finalSegments)
    gui.updateLog('Algorithm finished with {} final-size segments'.format(len(finalSegments)))

    return finalSegments, finalData, finalPaths

#--------------------------------------------------------------------------------------------------------------------------------------

#flags segments as background or foreground
def findBackground(finalData, maxVar, maxInt):

    gui = shareGui.getGui()
    gui.updateLog('Finding background segments background segments:')
    gui.updateLog('Using variance threshold of {}'.format(maxVar))
    gui.updateLog('Using intensity threshold of {}'.format(maxInt))

    background = numpy.zeros(len(finalData), dtype = int)

    #if the current segment has a variance and mean intensity below both of the threshold values, it is set to '1' in the background list,
    #indicating that the segment at that index is a background segment
    for n in range(len(finalData)):
        if(numpy.var(finalData[n]) < maxVar and numpy.mean(finalData[n]) < maxInt):
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

#--------------------------------------------------------------------------------------------------------------------------------------

#gets rid of all uneeded directories created during segmentation
#(currently, the pixel files are left alone for all final segments, in case they are needed, this could be changed later)
def cleanup(segmentDir, finalPaths):

    #create new directory to keep pixel info
    storagePath = '{}/pixelData/'.format(segmentDir)

    if not os.path.isdir(storagePath):
        try:
            os.makedirs(storagePath)
        except OSError:
            if not os.path.isdir(storagePath):
                raise

    #move pixel data of only final segments to the new directory
    for n in range(len(finalPaths)):
        path = finalPaths[n]
        shutil.move(path, '{}pixels_segment_{}'.format(storagePath, n))

    #remove everything else
    dirs = next(os.walk(segmentDir))[1]
    for dir in dirs:
        if dir.find('pixelData') == -1:
            shutil.rmtree('{}/{}'.format(segmentDir, dir))
    return


