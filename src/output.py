__author__ = 'hollowed'

import numpy
import math
import h5py
import shareGui
import pdb

def toMAPS(segments, backgrounds, dimensions, segmentDir):

    gui = shareGui.getGui()
    gui.updateLog('Exporting segment data to MAPS ROIs')
    imageSize = dimensions[0]*dimensions[1]
    file = h5py.File('{}/test.h5'.format(segmentDir), 'x')
    file.create_group('rois')

    fgSegments = segments[backgrounds==0] #take all indices from 'segments' where the corresponding index in 'backgrounds is 0
    bgSegments = segments[backgrounds==1]
    numFgFiles = int(math.ceil(float(len(fgSegments))/16.0)) #MAPS supports only 16 distinct ROI's in one dataset
    numBgFiles = int(math.ceil(float(len(bgSegments))/16.0)) #so this will find how mant datasets are needed to account for all segs
    fgRois = [numpy.zeros(imageSize, int) for _ in range(numFgFiles)] #array the size fo the image to be filled with ints 1-16
    bgRois = [numpy.zeros(imageSize, int) for _ in range(numBgFiles)]
    segs = [fgSegments, bgSegments]
    rois = [fgRois, bgRois]

    for i in range(len(rois)):
        regionNum = 1
        fileNum = 0
        for j in range(len(segs[i])):
            if(regionNum > 16):
                regionNum = 1
                fileNum += 1
            region = segs[i][j]
            for index in region:
                rois[i][fileNum][index] = regionNum
            regionNum += 1

        for n in range(len(rois[i])):
            nextData = rois[i][n].reshape(dimensions)
            if i==0: file.create_dataset('rois/fg_roi_{}'.format(n), data=nextData)
            elif i==1: file.create_dataset('rois/bg_roi_{}'.format(n), data=nextData)

    gui.updateLog('Saved ROIs to {}/test.h5'.format(segmentDir))

    return

