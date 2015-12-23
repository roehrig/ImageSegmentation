__author__ = 'hollowed'

import numpy
import math
import h5py
import shareGui
import pdb

def toMAPS(segments, backgrounds, dimensions, segmentDir, imageName):

    #takes all the final data gathered from finalize.py and turns it into hdf5 files, in the format that MAPS expected for ROIs

    gui = shareGui.getGui()
    gui.updateLog('Exporting segment data to MAPS ROIs')
    imageSize = dimensions[0]*dimensions[1]
    pixelIds = numpy.arange(imageSize)
    scanNumber = imageName.split('_')[1].split('.')[0]

    fgSegments = segments[backgrounds==0] #take all indices from 'segments' where the corresponding index in 'backgrounds is 0
    bgSegments = segments[backgrounds==1]
    numFgFiles = int(math.ceil(float(len(fgSegments))/16.0)) #MAPS supports only 16 distinct ROI's in one dataset
    numBgFiles = int(math.ceil(float(len(bgSegments))/16.0)) #so this will find how mant datasets are needed to account for all segs
    fgRois = [numpy.zeros(imageSize, int) for _ in range(numFgFiles)] #list of arrays, each the size of the image to be filled with ints 1-16
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

        f,b=0,0 #current foreground segment, current background segment
        for n in range(len(rois[i])):
            nextData = rois[i][n].reshape(dimensions) #next segment
            if i==0:
                file = h5py.File('{}/{}_roi_FG{}.h5'.format(segmentDir, scanNumber, f), 'w') #create a new hdf5 file (should be changed in the future, only one total is really needed
                file.create_dataset('MAPS_ROIS/fg_roi_{}'.format(n), data=nextData) #create a MAPS group, and add a dataset containing the next segment
                file.create_dataset('MAPS_ROIS/pixel_id', data=pixelIds)
                f+=1
            elif i==1:
                file = h5py.File('{}/{}_roi_BG{}.h5'.format(segmentDir, scanNumber, b), 'w')
                file.create_dataset('MAPS_ROIS/bg_roi_{}'.format(n), data=nextData)
                file.create_dataset('MAPS_ROIS/pixel_id', data=pixelIds)
                b+=1

    gui.updateLog('Saved ROIs to MAPS compatible HDF5 formats'.format(segmentDir))

    return

