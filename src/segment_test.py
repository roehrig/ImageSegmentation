__author__ = 'roehrig'

import os
from pixel import *
from imagedata import ImageFileData
from matrices import DiagonalMatrix as DM
from matrices import WeightMatrix as WM
from numpy import linalg as LA
from plotframe import ScatterPlot
from plotframe import HistogramPlot
import time
import shareGui
import pdb


def CalculateIntensitySigma(pixelList):

    mu = 0.0
    sum_squares = 0.0
    numPixels = len(pixelList)

    for pixel in pixelList:
        mu = mu + pixel.GetValue()

    mu = mu / numPixels

    for pixel in pixelList:
        sum_squares = sum_squares + pow(pixel.GetValue() - mu, 2)

    sigma = sum_squares / numPixels

    return sigma
#    return math.sqrt(sigma)


def CalculateLocationSigma(pixels):

    mu = 0.0
    sum_squares = 0.0
    numPixels = pixels.GetPixelArraySize()

    pixelArray = pixels.GetPixelArray()

    for pixel in pixelArray:
        mu = mu + pixel.GetLocation()

    mu = mu / numPixels

    for pixel in pixelArray:
        sum_squares = sum_squares + pow(pixel.GetLocation() - mu, 2)

    sigma = sum_squares / numPixels

    return sigma
#    return math.sqrt(sigma)


def DivideImage(secondVec, imageData, imageSize, datasize, locations, dividingValue):

    segmentOne = numpy.zeros(imageSize, dtype = tuple)
    segmentTwo = numpy.zeros(imageSize, dtype = tuple)
    posIndices = []
    negIndices = []
    numPos = 0
    numNeg = 0

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


    segmentInfo = {'segOne':segmentOne, 'segTwo':segmentTwo, 'posIndices':posIndices, 'negIndices':negIndices}


    return segmentInfo


def solveEigenVectors(diag, weight):

    finalMatrix = numpy.subtract(diag, weight)
    eigenValues, eigenVectors = LA.eig(finalMatrix)
    indices = eigenValues.argsort()
    secondVal = eigenValues[indices[1]]
    secondVec = eigenVectors[:, indices[1]]


    return (secondVec)


def SegmentImage (weightMatrix, data, image_dir, divideType, fileFormat, displayPlots, iterations): #now takes fileformat, displayPlots, and iterations

    gui = shareGui.getGui()                                                                         #added the gui object

    gui.updateLog("Creating diagonal matrix")
    diagonalMatrix = DM(data.width, data.height)
    diagonalMatrix.CreateMatrix(weightMatrix.GetMatrix())

    gui.updateLog('Solving for eigenvectors')
    secondVec = solveEigenVectors(diagonalMatrix.GetMatrix(), weightMatrix.GetMatrix())

    # Divide the image into two using the second smallest eigenvector.
    if divideType == 0:
        dividingValue = 0

    if divideType == 1:
        dividingValue = numpy.median(secondVec, axis=0)

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

    imageSize = data.GetImageSize()
    pixelLocations = numpy.arange(imageSize)
    imageData = data.GetImageData()

    gui.updateLog('Dividing image pixel values')
    segmentInfo = DivideImage(secondVec, imageData, imageSize, imageSize, pixelLocations, dividingValue)
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
        for i in range(numSteps - 1):

            dividingValue = dividingValue - stepSize
            segmentInfo = DivideImage(secondVec, imageData, imageSize, imageSize, pixelLocations, dividingValue)
            segmentOne = segmentInfo['segOne']
            segmentTwo = segmentInfo['segTwo']
            posIndices = segmentInfo['posIndices']
            negIndices = segmentInfo['negIndices']

            # Calculate the weights of the edges that were removed from the image.
            # Reduce the weight matrix to two new matrices, one for each image segment.
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
        gui.updateLog("Pixels in image one = %d" % len(posIndices))
        gui.updateLog("Pixels in image two = %d" % len(negIndices))
        gui.updateLog("Matrix one size = %dx%d" % (matrixOne.GetMatrix().shape[0], matrixOne.GetMatrix().shape[1]))
        gui.updateLog("Matrix two size = %dx%d" % (matrixTwo.GetMatrix().shape[0], matrixTwo.GetMatrix().shape[1]))

    # Create two arrays, each containing pixels from the original image, using
    # the indices returned from DivideImage.
    posPixels = numpy.take(imageData, posIndices, axis=0)
    negPixels = numpy.take(imageData, negIndices, axis=0)

    # Save the pixel locations for each new pixel array.
    posLocations = numpy.take(pixelLocations, posIndices)
    negLocations = numpy.take(pixelLocations, negIndices)

    cutNumber = 1
    image_path = image_dir + "/cut_" + str(cutNumber)
    pixel_path = image_dir + "/pixels_" + str(cutNumber)
    matrix_path = image_dir + "/matrices_" + str(cutNumber)

    if not os.path.isdir(image_path):
        try:
            os.makedirs(image_path)
            os.makedirs(matrix_path)
            os.makedirs(pixel_path)
        except OSError:
            if not os.path.isdir(image_path):
                raise
            if not os.path.isdir(matrix_path):
                raise
            if not os.path.isdir(pixel_path):
                raise

    filename = "/segment_1_"
    gui.updateLog("Writing image file {}1.{}".format(filename, fileFormat))
    data.WriteNewImage(segmentOne, '{}{}1.{}'.format(image_path,filename,fileFormat))     #this will now write to whatever the file format of the original is instead of just tif
    posArrays = {'pixels':posPixels, 'locations':posLocations}
    numpy.savez(pixel_path + filename + "1.npz", **posArrays)
    numpy.save(matrix_path + filename + "1.npy", matrixOne.GetMatrix())

    gui.updateLog("Writing image file {}2.{}".format(filename, fileFormat))
    data.WriteNewImage(segmentTwo, '{}{}2.{}'.format(image_path,filename,fileFormat))
    negArrays = {'pixels':negPixels, 'locations':negLocations}
    numpy.savez(pixel_path + filename + "2.npz", **negArrays)
    numpy.save(matrix_path + filename + "2.npy", matrixTwo.GetMatrix())
    gui.updateLog('Done')


    if(displayPlots):
        scatterPlotter = ScatterPlot("Original Image", secondVec)
        histogramPlotter = HistogramPlot("Original image", secondVec)

    gui.advanceProgressBar(60)

    while cutNumber < iterations:                                                       #The algorithm will now run for as many times as desired, with the variable 'iterations'
                                                                                        # given by the user in the gui
        gui.advanceProgressBar(40/iterations)
        imageNumber = 1
        cutNumber += 1

        image_path = image_dir + "/cut_" + str(cutNumber)
        pixel_path = image_dir + "/pixels_" + str(cutNumber)
        matrix_path = image_dir + "/matrices_" + str(cutNumber)

        prev_image_path = image_dir + "/cut_" + str(cutNumber - 1)
        prev_pixel_path = image_dir + "/pixels_" + str(cutNumber - 1)
        prev_matrix_path = image_dir + "/matrices_" + str(cutNumber - 1)

        if not os.path.isdir(image_path):
            try:
                os.makedirs(image_path)
                os.makedirs(matrix_path)
                os.makedirs(pixel_path)
            except OSError:
                if not os.path.isdir(image_path):
                    raise
                if not os.path.isdir(matrix_path):
                    raise
                if not os.path.isdir(pixel_path):
                    raise

        # Iterate through the image files and weight matrix files
        for image in os.listdir(prev_image_path):

            # Get the name of the file without the file extension.
            filename = os.path.splitext(image)[0]

            gui.updateLog("\nWorking on file %s" % image)
            # Create a new ImageFileData object and read in the image file.
            newData = ImageFileData(prev_image_path + "/" + image)
            newData.ReadImage()
            dimensions = newData.GetImageDimensions()
            gui.updateLog("Image width = %d, image height = %d" % dimensions)

            # Load the image pixels and pixel locations into arrays
            newPixelsArrays = numpy.load(prev_pixel_path + "/" + filename + ".npz")
            newPixels = newPixelsArrays['pixels']
            newLocations = newPixelsArrays['locations']
            gui.updateLog("Reading array of size %d from file %s.npz" % (newPixels.shape[0], filename))

            # Load the matrix data and create a new WeightMatrix
            temp = numpy.matrix(numpy.load(prev_matrix_path + "/" + filename + ".npy"))
            newWeightMatrix = WM(temp.shape[0], temp.shape[1])
            newWeightMatrix.SetMatrix(temp)
            gui.updateLog("Reading matrix of size %dx%d from file %s.npy" % (temp.shape[0], temp.shape[1], filename))


            # Create a new diagonal matrix.
            gui.updateLog("Creating diagonal matrix")
            diagonalMatrix = DM(len(newPixels), 1)                          #had to change .size to len() to count multi-dimensional array items properly
            diagonalMatrix.CreateMatrix(newWeightMatrix.GetMatrix())

            gui.updateLog('Solving for eigenvalues')
            secondVec = solveEigenVectors(diagonalMatrix.GetMatrix(), newWeightMatrix.GetMatrix())


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
            imageData = data.GetImageData()
            segmentInfo = DivideImage(secondVec, newPixels, imageSize, len(newPixels), newLocations, dividingValue)
            segmentOne = segmentInfo['segOne']
            segmentTwo = segmentInfo['segTwo']
            posIndices = segmentInfo['posIndices']
            negIndices = segmentInfo['negIndices']

            # Calculate the weights of the edges that were removed from the image.
            # Reduce the weight matrix to two new matrices, one for each image segment.
            cutSize, matrixOne, matrixTwo = newWeightMatrix.ReduceMatrix(posIndices, negIndices)
            gui.updateLog("Size of cut = %f" % cutSize)


            if divideType == 2:
                prevCutSize = cutSize
                prevSegInfo = segmentInfo
                for i in range(numSteps - 1):

                    dividingValue = dividingValue - stepSize
                    segmentInfo = DivideImage(secondVec, newPixels, imageSize, len(newPixels), newLocations, dividingValue)
                    segmentOne = segmentInfo['segOne']
                    segmentTwo = segmentInfo['segTwo']
                    posIndices = segmentInfo['posIndices']
                    negIndices = segmentInfo['negIndices']

                    # Calculate the weights of the edges that were removed from the image.
                    # Reduce the weight matrix to two new matrices, one for each image segment.
#                    edgeSum, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
                    cutSize, matrixOne, matrixTwo = newWeightMatrix.ReduceMatrix(posIndices, negIndices)
#                    print "Size of cut = %f" % cutSize
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
                gui.updateLog("Pixels in image one = %d" % len(posIndices))
                gui.updateLog("Pixels in image two = %d" % len(negIndices))
                gui.updateLog("Matrix one size = %dx%d" % (matrixOne.GetMatrix().shape[0], matrixOne.GetMatrix().shape[1]))
                gui.updateLog("Matrix two size = %dx%d" % (matrixTwo.GetMatrix().shape[0], matrixTwo.GetMatrix().shape[1]))


            # Create two arrays of pixels from the original image using
            # the indices returned from DivideImage.
            posPixels = numpy.take(newPixels, posIndices, axis=0)
            negPixels = numpy.take(newPixels, negIndices, axis=0)

            # Save the pixel locations for each new pixel array.
            posLocations = numpy.take(newLocations, posIndices)
            negLocations = numpy.take(newLocations, negIndices)

            if(displayPlots):
                scatterPlotter.AddPlot(image, secondVec)
                histogramPlotter.AddPlot(image, secondVec)                                     #histogram plots are now created on every iteration, and displayed in the gui results

            filename = "/segment_%d_%d" % (cutNumber, imageNumber)
            gui.updateLog("Writing image file {}.{}".format(filename, fileFormat))
            data.WriteNewImage(segmentOne, '{}{}.{}'.format(image_path, filename, fileFormat)) #this will now write to whatever the file format of the original is instead of just tif
            posArrays = {'pixels':posPixels, 'locations':posLocations}
            numpy.savez(pixel_path + "/%s.npz" % filename, **posArrays)
            numpy.save(matrix_path + "/%s.npy" % filename, matrixOne.GetMatrix())
            imageNumber += 1

            filename = "/segment_%d_%d" % (cutNumber, imageNumber)
            gui.updateLog("Writing image file {}.{}\n".format(filename, fileFormat))
            data.WriteNewImage(segmentTwo, '{}{}.{}'.format(image_path, filename, fileFormat))
            negArrays = {'pixels':negPixels, 'locations':negLocations}
            numpy.savez(pixel_path + "/%s.npz" % filename, **negArrays)
            numpy.save(matrix_path + "/%s.npy" % filename, matrixTwo.GetMatrix())

            imageNumber += 1


    if(displayPlots):                                                                          #Added this if statment to handle adding the scatter and histogram plots to the gui's
        scatterPlotter.displayPlots()                                                          #results frame
        histogramPlotter.displayPlots()

    return


# def mergeImages(imageList):                                                                    #This function if for marging many grayscale images into one multi-channel image, will revisit this later
#
#     allData = []
#     allSizes = []
#     firstImage = ImageFileData(imageList[0])
#     finalFormat = firstImage.GetFileFormat()
#     finalName = ('{}_merged'.format(imageList[0]))
#     mergedData = numpy.zeros((max(allSizes), len(imageList)))
#
#     for image in imageList:
#         data = ImageFileData(image)
#         allData.append(data.GetImageData())
#
#     for data1 in allData:
#         allSizes.append(data1.GetImageSize())
#         for data2 in allData:
#             if (data1.GetImageDimensions() != data2.GetImageDimensions()):
#                 answer = qt.QMessageBox.warning('Image dimensions are not equal. This will work, but all images will align at their TOP LEFT CORNERS, not their centers.', 'Ok', 'Cancel')
#                 if(answer == 1): return False
#             if(data1.GetImageFormat() != data2.GetImageFormat()):
#                 answer = qt.QMessageBox('Image formats are not the same. This will work, but the final image will be written in the format of the first image in the file list.', 'Ok', 'Cancel')
#                 if(answer == 1): return False
#
#     try:
#         try to merge the given images here using numpy.dstack()?
#
#     except:
#         qt.QMessageBox('Image merge failed.')
#         return False


def start(imagePath, divideType, maxPixelDistance, discretize, smoothValue, displayPlots, iterations):

    '''
    :param divideType = Set the type of dividing to be done.
        0 - divide the image using the value of zero
        1 - divide the image using the median value of the eignevector
        2 - try vector.size / (log(vector.size))^2 evenly spaced dividing points
    :param maxPixelDistance = how close 2 pixels must be to have a nonzero weight between them
    :param discretize = boolean whether or not to discretize
    :param smoothValue = iterations of smoothing (smoothing not called if smoothValue = 0)
    :param displayPlots = self explanatory boolean
    :param iterations = number of times to run the algorithm
    :return segmentDir or None
    '''

    gui = shareGui.getGui()
    allValid = True
    imageDir, imageFile = os.path.split(imagePath)
    segmentDir = '{}/{}_segmentation/{}'.format(imageDir, time.strftime("%m-%d-%Y"), imageFile)     #creates two directories titled as the current date > the original image name, to save all output
    dateDir = os.path.split(segmentDir)[0]
    suffix = 1
    totalSegs = 1
    for i in range(iterations):
        i+= 1
        totalSegs += i**2


    while(os.path.isdir(segmentDir) == True):                                                        #This while loop ensures no segmentations of a common image are overwritten
        segmentDir = '{}/{}_segmentation/{}_{}'.format(imageDir, time.strftime("%m-%d-%Y"), imageFile, suffix)
        suffix += 1

    #Creates the segmentation directory
    if not os.path.isdir(segmentDir):
        try:
            os.makedirs(segmentDir)
        except OSError:
            if not os.path.isdir(segmentDir):
                raise

    gui.updateLog('--- Starting ---\n')
    gui.updateLog('Creating segmentation directory at: %s' % segmentDir)
    gui.updateLog('Using base image located at: %s' % imagePath)
    gui.updateLog('Using divide type of %d' % divideType)
    gui.updateLog('Using maximum pixel distance of %d' % maxPixelDistance)
    gui.updateLog('Algorithm with make %d cuts (%d total segments)' % (iterations, totalSegs))

    data = ImageFileData(imagePath, segmentDir)
    gui.updateLog('\n--- Reading Image ---\n')
    data.ReadImage()

    if smoothValue > 0:
        data.SmoothImage(smoothValue)
    if discretize == True:
        #Discretizes the image, using 255 (white) as the maximum value
        #If the image is not grayscale and cannot be discretized, the user can either continue (allValid = True), or abort (allValid = false)
        allValid = data.DiscretizeImage(255, 0)

    if allValid:
        gui.advanceProgressBar(10)
        imageData = data.GetImageData()
        imageSize = data.GetImageSize()
        dimensions = data.GetImageDimensions()
        channels = data.GetChannels()
        fileFormat = data.GetFileFormat()
        # Create an array of pixel locations, location=sqrt(x^2 + y^2)
        locationValues = data.pixels.CreateLocationArray()
        sigmaI = numpy.var(imageData)
        sigmaX = numpy.var(locationValues)

        gui.updateLog("Image mode is %s" % data.GetImageMode())
        gui.updateLog("Image format is %s" % fileFormat)
        gui.updateLog("Number of image pixels = %d" % imageSize)
        gui.updateLog("Image width = %d, image height = %d" % dimensions)
        gui.updateLog("Intensity variance = %f" % sigmaI)
        gui.updateLog("Location variance = %f" % sigmaX)
        gui.advanceProgressBar(20)                                                                      #All of these lines change the graphic of the gui's progress bar


        gui.updateLog("\n--- Creating weight matrix ---\n")
        weightMatrix = WM(data.size, data.size)
        weightMatrix.SetPixelData(data.GetPixels(), maxPixelDistance)

        t0 = time.time()
        weightMatrix.CreateMatrix(sigmaI, sigmaX)
        t2 = time.time() - t0
        gui.updateLog('Parallel building of weight matrix took {} seconds'.format(t2))
        gui.advanceProgressBar(50)


        gui.updateLog('\n--- Starting segmentation ---\n')
        SegmentImage(weightMatrix, data, segmentDir, divideType, fileFormat, displayPlots, iterations) #now passes fileformat, displayPlots, and the desired number of iterations
        gui.advanceProgressBar(100)
        gui.updateLog('--- Segmentation completed. Click the results icon in the toolbar to view segments and plots. ---')

        gui.setSegmentPath(segmentDir)
        gui.setRawData(list(imageData))

    else:
        #If gui recieves None instead of segmentDir, it knows the segmentation was aborted
        #In that case, epmty segmentation directory created above is deleted.
        #If the current image segentation directory is the only directory in the date folder (current_date_segmentation/image.ext_dir)
        # it deletes the entire directory, otherwise just deletes the current image segmentation direcctory

        os.rmdir(segmentDir)
        try:
            os.rmdir(dateDir)
        except OSError:
            #meaning there are other image segmentations saved in the date directoy
            pass

        return None
