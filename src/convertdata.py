__author__ = 'hollowed'

import pdb
from PIL import Image
import numpy
import os

#--------------------------------------------------------------------------------------------------------------------------------------

def toArray(path, file, exchange, channels, channelNames):

    #takes a HDF5 or h5 file and extracts its data as a numpy array (or mulitple, if multiple channels are selected)
    #this is run immediately upon opneing an image, not when the segmentation is run,
    #in order to gather unit data for parameter selection

    path, filename = os.path.split(path)
    filename = filename.split('.')[0]
    path = '{}/{}'.format(path, filename)
    dataStr = '{}/data'.format(exchange)
    channelStr = '{}/channel_names'.format(exchange)
    allData = file[dataStr] #the XRF data from the hdf file
    allChannels = file[channelStr] #the channel names from the hdf file
    width, height = allData.shape[2], allData.shape[1]
    imageSize = width*height
    dimensions = (width, height)
    reducedData = numpy.empty((len(channels), imageSize))

    #gather only the (2D) arrays within the (3D) data array corresponding to the selected channels
    i=0
    for n in range(len(allData)):
        if n in channels and allChannels[n] == channelNames[channels.index(n)]:
            reducedData[i] = numpy.ravel(allData[n])
            i+=1

    #current image must not have had the channels specified
    if i != len(channels):
        raise KeyError

    #save each "image" (array of XRF "pixel" data) as a numpy file
    #that contains the data as the first entry, and the channel name as the second
    dataPaths = []
    imagePaths = []

    for n in range(len(reducedData)):
        dataPath = '{}_{}.npz'.format(path, numpy.array(channelNames[n]))
        saveData = {'data':reducedData[n], 'channels':numpy.array(channelNames[n]), 'dimensions':dimensions}
        numpy.savez(dataPath, **saveData)
        dataPaths.append(dataPath)

        #render and save grayscale image
        imageData = reducedData[n]
        imagePath = toImage(imageData, path, channelNames, width, height, 'L')
        imagePaths.append(imagePath)

    return dataPaths, imagePaths

#--------------------------------------------------------------------------------------------------------------------------------------

def toStackedArray(path, file, exchange, channels, channelNames):

    #does the same as above, but if multiple channels are selected, one numpy array of tuples will be generated,
    #rather than several numpy arrays

    path, filename = os.path.split(path)
    filename = filename.split('.')[0]
    path = '{}/{}'.format(path, filename)
    dataStr = '{}/data'.format(exchange)
    channelStr = '{}/channel_names'.format(exchange)
    allChannels = file[channelStr] #the channel names from the hdf file
    allData = file[dataStr] #the XRF data from the hdf file
    width, height = allData.shape[2], allData.shape[1]
    imageSize = width*height
    dimensions = (width, height)
    reducedData = numpy.empty((len(channels), imageSize))

    #gather only the (2D) arrays within the (3D) data array corresponding to the selected channels
    i=0
    for n in range(len(allData)):
        if n in channels and allChannels[n] == channelNames[channels.index(n)]:
            reducedData[i] = numpy.ravel(allData[n])
            i+=1

    #current image must not have had the channels specified
    if i != len(channels):
        raise KeyError

    #save all selected channels as one 3D numpy array
    #save this array as the first entry of saveData, and save the channel names as the second entry
    stackedImageData = numpy.swapaxes(reducedData, 1, 0)
    stackedData = map(tuple, stackedImageData)
    dataPath = '{}_{}.npz'.format(path, channelNames)
    saveData = {'data':stackedData, 'channels':channelNames, 'dimensions':dimensions}
    numpy.savez(dataPath, **saveData)

    #render and save RGB image
    imagePath = toImage(stackedImageData, path, channelNames, width, height, 'RGB')

    return dataPath, imagePath

#--------------------------------------------------------------------------------------------------------------------------------------

def toImage(data, path, channelNames, width, height, mode):

    #rendering an image from the data by scaling to a 255 maximum.
    #The percise floating point data found to toArray() or stackChannels() is still
    #used for the segmentation algorithm, this image will simply be used to display
    # on the results page of the gui

    #If the data has multiple channels (toImage() was called from stackChannels()), at most the first
    #THREE channels of the data are used to render an image. This is because what we really want is the ROI
    #data, so this image will just be for qualitative display within this program. Also, mainly beacause PIL
    #supports no image modes with more than 3 additive color channels (RGB).

    #multiplication is faster, so is used for scaling, but if a divide by zero error occurs,
    # division is used (unlikely, implies blank channel)

    if(mode == 'RGB'):
        #remove all but first three channel values, cast pixels to tuples
        imageData = numpy.zeros((width*height, 3))
        for n in range(len(data)):
            imageData[n][:len(data[n])] = data[n][:3]
        try: imageData = (imageData * (255.0/imageData.max())).astype(int)
        except ZeroDivisionError: imageData = imageData / (imageData.max()/255.0)
        imageData = map(tuple, imageData)
    else:
        try: imageData = (data * (255.0/data.max())).astype(int)
        except ZeroDivisionError: imageData = data / (data.max()/255.0)

    imagePath = '{}_{}_image.png'.format(path, channelNames)
    img = Image.new(mode, (width, height))
    img.putdata(imageData)
    img.save(imagePath)
    return imagePath