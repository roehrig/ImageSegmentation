__author__ = 'roehrig'

import os
from pixel import *
from imagedata import ImageFileData, ImageArrayData
from matrices import DiagonalMatrix as DM
from matrices import WeightMatrix as WM
from numpy import linalg as LA
from plotframe import ScatterPlot
from plotframe import HistogramPlot
import time
import shareGui
import pdb

#--------------------------------------------------------------------------------------------------------------------------------------

#Creates the segmentation save destination
def definePath(cutNumber, image_dir):

    pixel_path = image_dir + "/pixels_" + str(cutNumber)
    matrix_path = image_dir + "/matrices_" + str(cutNumber)

    prev_pixel_path = image_dir + "/pixels_" + str(cutNumber - 1)
    prev_matrix_path = image_dir + "/matrices_" + str(cutNumber - 1)

    if not os.path.isdir(pixel_path):
        try:
            os.makedirs(matrix_path)
            os.makedirs(pixel_path)
        except OSError:
            if not os.path.isdir(matrix_path):
                raise
            if not os.path.isdir(pixel_path):
                raise

    paths = [pixel_path, matrix_path, prev_pixel_path, prev_matrix_path]
    return paths

#--------------------------------------------------------------------------------------------------------------------------------------

#subtracts the weight matrix from the diagonal weight matrix, solves for eigenvectors, and returns the second eigenvector
def solveEigenVectors(diag, weight):

    finalMatrix = numpy.subtract(diag, weight)
    eigenValues, eigenVectors = LA.eig(finalMatrix)
    indices = eigenValues.argsort()
    secondVal = eigenValues[indices[1]]
    secondVec = eigenVectors[:, indices[1]]

    return (secondVec)

#--------------------------------------------------------------------------------------------------------------------------------------

#Calculates each normalized cut
def DivideImage(secondVec, imageData, imageSize, datasize, locations, dividingValue):

    #uses tuple as type to support multi-layered images
    segmentOne = numpy.zeros(imageSize, dtype = tuple)
    segmentTwo = numpy.zeros(imageSize, dtype = tuple)
    posIndices = []
    negIndices = []
    numPos = 0
    numNeg = 0

    #inspect each pixel and place it on one side of the cut
    for i in range(datasize):
        if secondVec[i] >= dividingValue:
            numPos += 1
            try:
                segmentOne[locations[i]] = tuple(imageData[i])
            except TypeError:
                segmentOne[locations[i]] = imageData[i]
            posIndices.append(i)
        else:
            numNeg += 1
            try:
                segmentTwo[locations[i]] = tuple(imageData[i])
            except TypeError:
                segmentTwo[locations[i]] = imageData[i]
            negIndices.append(i)

    #return the contents of each segment
    segmentInfo = {'segOne':segmentOne, 'segTwo':segmentTwo, 'posIndices':posIndices, 'negIndices':negIndices}


    return segmentInfo

#--------------------------------------------------------------------------------------------------------------------------------------

#Preforms the algorithm
def workSegment(haltThreshold, weightMatrix, data, divideType, displayPlots, cutNumber, paths, imageNumber, image):

    gui = shareGui.getGui()
    pixel_path, matrix_path = paths[0], paths[1]
    prev_pixel_path, prev_matrix_path = paths[2], paths[3]
    imageSize = data.GetImageSize()
    imageData = data.GetImageData()
    pixelLocations = numpy.arange(imageSize)

    #initial segmentation-----------------------
    if (cutNumber == 1):
        pixelLength = imageSize
        divideData = imageData

    #each subsequent segmentation---------------
    else:
        # Get the name of the file without the file extension.
        filename = os.path.splitext(image)[0]

        gui.updateLog("\nWorking on %s" % image)

        # Load the image pixels and pixel locations into arrays
        newPixelsArrays = numpy.load(prev_pixel_path + "/" + filename + ".npz")
        newPixels = newPixelsArrays['pixels']
        divideData = newPixels
        pixelLocations = newPixelsArrays['locations']
        pixelLength = len(newPixels)
        gui.updateLog("Reading array of size %d from file %s.npz" % (newPixels.shape[0], filename))
        #If the size of the loaded image is below the halting threshold, do not segment, and return (ending this branch of the segmentation)

        if(len(newPixels) < haltThreshold):
            gui.updateLog('Skipping file %s: segment size is below specified halting threshold.' % image)
            #return value of '0' means the segmentation on this branch has ended
            return(0)

        # Load the matrix data and create a new WeightMatrix
        temp = numpy.matrix(numpy.load(prev_matrix_path + "/" + filename + ".npy"))
        weightMatrix = WM(temp.shape[0], temp.shape[1])
        weightMatrix.SetMatrix(temp)
        gui.updateLog("Reading matrix of size %dx%d from file %s.npy" % (temp.shape[0], temp.shape[1], filename))

    #Create a new diagonal matrix.
    gui.updateLog("Creating diagonal matrix")
    if (cutNumber > 1):
        diagonalMatrix = DM(len(newPixels), 1)                          #had to change .size to len() to count multi-dimensional array items properly
    else:
        diagonalMatrix = DM(data.width, data.height)
    diagonalMatrix.CreateMatrix(weightMatrix.GetMatrix())

    gui.updateLog('Solving for eigenvalues')
    secondVec = solveEigenVectors(diagonalMatrix.GetMatrix(), weightMatrix.GetMatrix())


    # Divide the image into two using the second smallest eigenvector.
    if divideType == 0:
        dividingValue = 0
        numsteps = 1

    if divideType == 1:
        dividingValue = numpy.median(secondVec, axis=0)
        numsteps = 1

    if divideType == 2:
        numSteps = int(secondVec.size / pow(math.log10(secondVec.size), 2))
        stepSize = (numpy.amax(secondVec) - numpy.amin(secondVec)) / numSteps
        if numSteps < 2:
            numSteps = 1
            dividingValue = numpy.median(secondVec, axis=0)
        else:
            maxVal = numpy.amax(secondVec)
            minVal = numpy.amin(secondVec)
            stepSize = (maxVal - minVal) / numSteps
            dividingValue = maxVal

    gui.updateLog('Dividing image pixel values')
    segmentInfo = DivideImage(secondVec, divideData, imageSize, pixelLength, pixelLocations, dividingValue)
    segmentOne = segmentInfo['segOne']
    segmentTwo = segmentInfo['segTwo']
    posIndices = segmentInfo['posIndices']
    negIndices = segmentInfo['negIndices']

    # Calculate the weights of the edges that were removed from the image.
    # Reduce the weight matrix to two new matrices, one for each image segment.
    cutSize, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
    gui.updateLog("Size of cut = %f" % cutSize)


    if divideType == 2:
        prevCutSize = cutSize
        prevSegInfo = segmentInfo
        prevMatrixOne = matrixOne
        prevMatrixTwo = matrixTwo

        #continue until smallest cut is found
        for i in range(numSteps - 1):
            dividingValue = dividingValue - stepSize
            segmentInfo = DivideImage(secondVec, divideData, imageSize, pixelLength, pixelLocations, dividingValue)
            segmentOne = segmentInfo['segOne']
            segmentTwo = segmentInfo['segTwo']
            posIndices = segmentInfo['posIndices']
            negIndices = segmentInfo['negIndices']

            # Calculate the weights of the edges that were removed from the image.
            # Reduce the weight matrix to two new matrices, one for each image segment.
            #edgeSum, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
            cutSize, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)

            if cutSize < prevCutSize:
                prevCutSize = cutSize
                prevMatrixOne = matrixOne
                prevMatrixTwo = matrixTwo
                prevSegInfo = segmentInfo

        cutSize = prevCutSize
        matrixOne = prevMatrixOne
        matrixTwo = prevMatrixTwo
        gui.updateLog("Size of choosen cut = %f" % cutSize)
        segmentInfo = prevSegInfo
        segmentOne = segmentInfo['segOne']
        segmentTwo = segmentInfo['segTwo']
        posIndices = segmentInfo['posIndices']
        negIndices = segmentInfo['negIndices']

    #Progress output to gui.
    gui.updateLog("Pixels in segment one = %d" % len(posIndices))
    gui.updateLog("Pixels in segment two = %d" % len(negIndices))
    gui.updateLog("Matrix one size = %dx%d" % (matrixOne.GetMatrix().shape[0], matrixOne.GetMatrix().shape[1]))
    gui.updateLog("Matrix two size = %dx%d" % (matrixTwo.GetMatrix().shape[0], matrixTwo.GetMatrix().shape[1]))

    # Create two arrays of pixels from the original image using
    # the indices returned from DivideImage.
    posPixels = numpy.take(divideData, posIndices, axis=0)
    negPixels = numpy.take(divideData, negIndices, axis=0)


    #Final output to image files
    #Segment One-----------------------------------------------
    filename = "/segment_%d_%d" % (cutNumber, imageNumber)
    # Save the pixel locations for each new pixel array.
    posLocations = numpy.take(pixelLocations, posIndices)
    gui.updateLog("Writing pixels file {}.npz".format(filename))
    posArrays = {'pixels':posPixels, 'locations':posLocations}
    numpy.savez(pixel_path + "/%s.npz" % filename, **posArrays)
    gui.updateLog("Writing matrix file {}.npy".format(filename))
    numpy.save(matrix_path + "/%s.npy" % filename, matrixOne.GetMatrix())

    imageNumber += 1

    #Segment Two-----------------------------------------------
    filename = "/segment_%d_%d" % (cutNumber, imageNumber)
    negLocations = numpy.take(pixelLocations, negIndices)
    gui.updateLog("Writing pixels file {}.npz".format(filename))
    negArrays = {'pixels':negPixels, 'locations':negLocations}
    numpy.savez(pixel_path + "/%s.npz" % filename, **negArrays)
    gui.updateLog("Writing matrix file {}.npy".format(filename))
    numpy.save(matrix_path + "/%s.npy" % filename, matrixTwo.GetMatrix())


    #Displays scatter plots and histograms in the results panel in the gui if selected (probably to be removed)
    if(displayPlots):
        scatPlt = ScatterPlot("Original Image", secondVec)
        histPlt = HistogramPlot("Original image", secondVec)
        if(cutNumber > 1):
            scatPlt.AddPlot(image, secondVec)
            histPlt.AddPlot(image, secondVec)

    #return value of '1' means the segmentation is still going on this branch (the current working segment will be segmented again)
    return(1)

#--------------------------------------------------------------------------------------------------------------------------------------

#Starts the algorithm, repeats it for each new segment
def SegmentImage (weightMatrix, data, image_dir, divideType, displayPlots, haltThreshold, numImages):

    #-------------------- First segment --------------------
    cutNumber = 1
    imageNumber = 1
    #'branches' is a list of 1's and 0's, with a zero in the index of a branch-ending segment (a segment size that has reached
    # the haltThreshold and will not be cut any further). At the end of segmentation, all segments should be a 0.
    #'currentBranches is a list of the images in the current working directory (set of segments from previous cut). If all
    # these are 0, the algorithm knows to stop
    branches = []
    currentBranches = [1] # = [1] simply to start while loop
    #define path for output
    paths = definePath(cutNumber, image_dir)

    workSegment(haltThreshold, weightMatrix, data, divideType, displayPlots, cutNumber, paths, imageNumber, None)

    #-------------------- Subsequent segments --------------------

    #while there is still an ongoing branch (with a segment size above the halting threshold), continue segmentation on those branches
    while (numpy.sum(currentBranches) > 0):

        imageNumber = 1
        cutNumber += 1
        #update path for output
        paths = definePath(cutNumber, image_dir)
        #preallocate space for currentBranches as large as the number of active segments left from the last cut (active = SegSize>haltThresh)
        currentBranches = numpy.zeros(len(os.listdir(paths[3])), dtype = int)
        i = 0

        # Iterate through the image files and weight matrix files of the previous cut
        for file in os.listdir(paths[2]):
            currentBranches[i] = workSegment(haltThreshold, None, data, divideType, displayPlots, cutNumber, paths, imageNumber, file)
            imageNumber += 2
            i += 1
        #add contents of currentBranch to branches
        branches.extend(currentBranches)

    return(branches)

#--------------------------------------------------------------------------------------------------------------------------------------

#initiates the segmentation; makes directories, loads image data, builds weight matrix, etc.
def start(imagePath, divideType, maxPixelDistance, smoothValue, displayPlots, haltThreshold, numImages):

    '''
    :param divideType = Set the type of dividing to be done.
        0 - divide the image using the value of zero
        1 - divide the image using the median value of the eignevector
        2 - try vector.size / (log(vector.size))^2 evenly spaced dividing points
    :param maxPixelDistance = how close 2 pixels must be to have a nonzero weight between them
    :param smoothValue = iterations of smoothing (smoothing not called if smoothValue = 0)
    :param displayPlots = self explanatory boolean
    :param haltThreshold = this is the pixel size that will end the cutting of a segment
    :return segmentDir or None
    '''

    gui = shareGui.getGui()
    imageDir, imageFile = os.path.split(imagePath)
    imageName, extension = imageFile.split('.')
    #creates two directories titled as the current date > the original image name, to save all output
    segmentDir = '{}/{}_segmentation/{}'.format(imageDir, time.strftime("%m-%d-%Y"), imageName)
    dateDir = os.path.split(segmentDir)[0]
    suffix = 1

    #This while loop ensures no segmentations of a common image are overwritten
    while(os.path.isdir(segmentDir) == True):
        segmentDir = '{}/{}_segmentation/{}_{}'.format(imageDir, time.strftime("%m-%d-%Y"), imageName, suffix)
        suffix += 1

    #Creates the segmentation directory
    if not os.path.isdir(segmentDir):
        try:
            os.makedirs(segmentDir)
        except OSError:
            if not os.path.isdir(segmentDir):
                raise

    gui.updateLog('------------------------ Starting on image {} ------------------------\n'.format(imageFile))
    gui.updateLog('Creating segmentation directory at: %s' % segmentDir)
    gui.updateLog('Using image located at: %s' % imagePath)
    gui.updateLog('Using divide type of %d' % divideType)
    gui.updateLog('Using maximum pixel distance of %d' % maxPixelDistance)
    gui.updateLog('Using halting threshold of %d' % haltThreshold)

    #loads selected image into imagedata object, depending on whether the user
    # selected data is a numpy array file (user opened a hdf5), or a nromal image file
    if extension == 'npz':
        data = ImageArrayData(imagePath, segmentDir)
    else:
        data = ImageFileData(imagePath, segmentDir)
    gui.updateLog('\n--- Reading Image ---\n')
    data.ReadImage()

    if smoothValue > 0:
        data.SmoothImage(smoothValue)

    #Setup
    gui.advanceProgressBar(10/numImages)
    imageData = data.GetImageData()
    imageSize = data.GetImageSize()
    dimensions = data.GetImageDimensions()
    channels = data.GetChannels()
    fileFormat = data.GetFileFormat()
    # Create an array of pixel locations, location=sqrt(x^2 + y^2)
    locationValues = data.pixels.CreateLocationArray()
    sigmaI = numpy.var(imageData)
    sigmaX = numpy.var(locationValues)

    #Output image properties
    gui.updateLog("Image mode is %s" % data.GetImageMode())
    gui.updateLog("Data format is %s" % fileFormat)
    gui.updateLog("Data channels are: {}".format(channels))
    gui.updateLog("Number of image pixels = %d" % imageSize)
    gui.updateLog("Image width = %d, image height = %d" % dimensions)
    gui.updateLog("Intensity variance = %f" % sigmaI)
    gui.updateLog("Location variance = %f" % sigmaX)
    #All of these lines change the graphic of the gui's progress bar
    gui.advanceProgressBar(10/numImages)

    #create weight matrix
    gui.updateLog("\n--- Creating weight matrix ---\n")
    weightMatrix = WM(data.size, data.size)
    weightMatrix.SetPixelData(data.GetPixels(), maxPixelDistance)

    #time weight matrix build
    t0 = time.time()
    weightMatrix.CreateMatrix(sigmaI, sigmaX)
    t2 = '%.2f' % (time.time() - t0)
    gui.updateLog('Parallel building of weight matrix took {} seconds'.format(t2))
    gui.advanceProgressBar(30/numImages)

    #Starts segmentation
    gui.updateLog('\n--- Starting segmentation---\n')
    t0 = time.time()
    branches = SegmentImage(weightMatrix, data, segmentDir, divideType, displayPlots, haltThreshold, numImages)
    t2 = '%.2f' % (time.time() - t0)
    gui.updateLog('\n\n--- Segmentation completed (took {} seconds) ---\n\n'.format(t2))

    gui.setSegmentPath(segmentDir)
    gui.setRawData(list(imageData))
    #This is the ultimate return back to the gui, a list of data resultant from the segmentation, for the gui to then pass to finalize.py and finalize
    segmentData = [branches, segmentDir, data, dimensions]
    return(segmentData)

