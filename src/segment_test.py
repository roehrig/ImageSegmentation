__author__ = 'roehrig'

import numpy
import os
from PIL import Image
from pixel import *
from imagedata import ImageFileData
from matrices import DiagonalMatrix as DM
from matrices import WeightMatrix as WM
from numpy import linalg as LA
from plotframe import ScatterPlot
from plotframe import HistogramPlot

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

def DivideImage(secondVec, imageData, imageSize, datasize, locations, dividingValue, channels):

    segmentOne = numpy.zeros(imageSize, dtype=tuple)  #instead of creating an array of ints, this now creates an array of tuples
    segmentTwo = numpy.zeros(imageSize, dtype=tuple)  #with its each tuples length corresonding to the # of channels in the image
                                                                #'channels' is gotten from ImageData.GetBands() and passed through SegmentImage()
    posIndices = []
    negIndices = []

#   print "Segmenting image"
    numPos = 0
    numNeg = 0
    for i in range(datasize):
        if secondVec[i] >= dividingValue:
            numPos += 1
            segmentOne[locations[i]] = imageData[i]
            posIndices.append(i)
        else:
            numNeg += 1
            segmentTwo[locations[i]] = imageData[i]
            negIndices.append(i)

#    print "Number of pixels in segment one = %d" % numPos
#    print "Number of pixels in segment two = %d" % numNeg

    segmentInfo = {'segOne':segmentOne, 'segTwo':segmentTwo, 'posIndices':posIndices, 'negIndices':negIndices}

    return segmentInfo

def SegmentImage (weightMatrix, data, image_dir, divideType, channels, fileFormat): #now takes channels and fileformat

    print "Creating diagonal matrix"
    diagonalMatrix = DM(data.width, data.height)
    diagonalMatrix.CreateMatrix(weightMatrix.GetMatrix())

    print "Calculating D-W"
    finalMatrix = numpy.subtract(diagonalMatrix.GetMatrix(), weightMatrix.GetMatrix())

    print "Solving for eigenvalues"
    eigenValues, eigenVectors = LA.eig(finalMatrix)
    indices = eigenValues.argsort()
    secondVal = eigenValues[indices[1]]
    secondVec = eigenVectors[:, indices[1]]

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

    segmentInfo = DivideImage(secondVec, imageData, imageSize, imageSize, pixelLocations, dividingValue, channels) #now passes channels
    segmentOne = segmentInfo['segOne']
    segmentTwo = segmentInfo['segTwo']
    posIndices = segmentInfo['posIndices']
    negIndices = segmentInfo['negIndices']

    # Calculate the weights of the edges that were removed from the image.
    # Reduce the weight matrix to two new matrices, one for each image segment.
#    edgeSum, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
    cutSize, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
#    matrixOneSum = numpy.sum(matrixOne.GetMatrix(), dtype=numpy.float64)
#    matrixTwoSum = numpy.sum(matrixTwo.GetMatrix(), dtype=numpy.float64)
#    cutSize = (edgeSum / matrixOneSum) + (edgeSum / matrixTwoSum)
#    print "Size of cut = %f" % cutSize

    if divideType == 2:
#        prevSum = edgeSum
#        prevMatrixOneSum = matrixOneSum
#        prevMatrixTwoSum = matrixTwoSum
        prevCutSize = cutSize
        prevSegInfo = segmentInfo
        for i in range(numSteps - 1):

            dividingValue = dividingValue - stepSize
            segmentInfo = DivideImage(secondVec, imageData, imageSize, imageSize, pixelLocations, dividingValue, channels) #now passes channels
            segmentOne = segmentInfo['segOne']
            segmentTwo = segmentInfo['segTwo']
            posIndices = segmentInfo['posIndices']
            negIndices = segmentInfo['negIndices']

            # Calculate the weights of the edges that were removed from the image.
            # Reduce the weight matrix to two new matrices, one for each image segment.
#            edgeSum, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
            cutSize, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
#            matrixOneSum = numpy.sum(matrixOne.GetMatrix(), dtype=numpy.float64)
#            matrixTwoSum = numpy.sum(matrixTwo.GetMatrix(), dtype=numpy.float64)
#            cutSize = (edgeSum / matrixOneSum) + (edgeSum / matrixTwoSum)
#            print "Size of cut = %f" % cutSize
            if cutSize < prevCutSize:
#                prevSum = edgeSum
                prevCutSize = cutSize
                prevMatrixOne = matrixOne
                prevMatrixTwo = matrixTwo
                prevSegInfo = segmentInfo

#        edgeSum = prevSum
        cutSize = prevCutSize
        matrixOne = prevMatrixOne
        matrixTwo = prevMatrixTwo
#        matrixOneSum = numpy.sum(matrixOne.GetMatrix(), dtype=numpy.float64)
#        matrixTwoSum = numpy.sum(matrixTwo.GetMatrix(), dtype=numpy.float64)
#        cutSize = (edgeSum / matrixOneSum) + (edgeSum / matrixTwoSum)
        print "Size of choosen cut = %f" % cutSize
        segmentInfo = prevSegInfo
        segmentOne = segmentInfo['segOne']
        segmentTwo = segmentInfo['segTwo']
        posIndices = segmentInfo['posIndices']
        negIndices = segmentInfo['negIndices']
        print "Pixels in image one = %d" % len(posIndices)
        print "Pixels in image two = %d" % len(negIndices)
        print "Matrix one size = %dx%d" % (matrixOne.GetMatrix().shape[0], matrixOne.GetMatrix().shape[1])
        print "Matrix two size = %dx%d" % (matrixTwo.GetMatrix().shape[0], matrixTwo.GetMatrix().shape[1])

    # Create two arrays, each containing pixels from the original image, using
    # the indices returned from DivideImage.
    posPixels = numpy.take(imageData, posIndices)
    negPixels = numpy.take(imageData, negIndices)

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
    print "Writing image file %s_1.tif" % filename
    #print("data in segment 1: {}".format(segmentOne)) #for debug
    data.WriteNewImage(segmentOne, image_path + filename +"1." + fileFormat) #this will now write to whatever the file format of the original is instead of just tif
    print "Writing image file %s_2.tif\n" % filename
    data.WriteNewImage(segmentTwo, image_path + filename + "2." + fileFormat)
    posArrays = {'pixels':posPixels, 'locations':posLocations}
    numpy.savez(pixel_path + filename + "1.npz", **posArrays)
    negArrays = {'pixels':negPixels, 'locations':negLocations}
    numpy.savez(pixel_path + filename + "2.npz", **negArrays)
    numpy.save(matrix_path + filename + "1.npy", matrixOne.GetMatrix())
    numpy.save(matrix_path + filename + "2.npy", matrixTwo.GetMatrix())

#!    plotter = ScatterPlot("Original Image", secondVec)
#!    plotter = HistogramPlot("original image", secondVec)

    while cutNumber < 2:
        imageNumber = 1
        cutNumber = cutNumber + 1

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

            print "Working on file %s" % image
            # Create a new ImageFileData object and read in the image file.
            newData = ImageFileData(prev_image_path + "/" + image)
            newData.ReadImage()
            dimensions = newData.GetImageDimensions()
            print "Image width = %d, image height = %d" % dimensions

            # Load the image pixels and pixel locations into arrays
            newPixelsArrays = numpy.load(prev_pixel_path + "/" + filename + ".npz")
            newPixels = newPixelsArrays['pixels']
            print "Reading array of size %d from file %s.npz" % (newPixels.shape[0], filename)
            newLocations = newPixelsArrays['locations']

            # Load the matrix data and create a new WeightMatrix
            temp = numpy.matrix(numpy.load(prev_matrix_path + "/" + filename + ".npy"))
            print "Reading matrix of size %dx%d from file %s.npy" % (temp.shape[0], temp.shape[1], filename)
            newWeightMatrix = WM(temp.shape[0], temp.shape[1])
            newWeightMatrix.SetMatrix(temp)

            # Create a new diagonal matrix.
#            print "Creating diagonal matrix"
            diagonalMatrix = DM(newPixels.size, 1)
            diagonalMatrix.CreateMatrix(newWeightMatrix.GetMatrix())

#            print "Calculating D-W"
            finalMatrix = numpy.subtract(diagonalMatrix.GetMatrix(), newWeightMatrix.GetMatrix())

#            print "Solving for eigenvalues"
            eigenValues, eigenVectors = LA.eig(finalMatrix)
            indices = eigenValues.argsort()
            secondVal = eigenValues[indices[1]]
            secondVec = eigenVectors[:, indices[1]]

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

            imageData = data.GetImageData()
            segmentInfo = DivideImage(secondVec, newPixels, imageSize, newPixels.size, newLocations, dividingValue, channels) #now passes channels
            segmentOne = segmentInfo['segOne']
            segmentTwo = segmentInfo['segTwo']
            posIndices = segmentInfo['posIndices']
            negIndices = segmentInfo['negIndices']

            # Calculate the weights of the edges that were removed from the image.
            # Reduce the weight matrix to two new matrices, one for each image segment.
#            edgeSum, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
            cutSize, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
#            matrixOneSum = numpy.sum(matrixOne.GetMatrix(), dtype=numpy.float64)
#            matrixTwoSum = numpy.sum(matrixTwo.GetMatrix(), dtype=numpy.float64)
#            cutSize = (edgeSum / matrixOneSum) + (edgeSum / matrixTwoSum)
#            print "Size of cut = %f" % cutSize

            if divideType == 2:
#                prevSum = edgeSum
#                prevMatrixOneSum = matrixOneSum
#                prevMatrixTwoSum = matrixTwoSum
                prevCutSize = cutSize
#                print "Size of cut = %f" % prevCutSize
                prevSegInfo = segmentInfo
                for i in range(numSteps - 1):

                    dividingValue = dividingValue - stepSize
                    segmentInfo = DivideImage(secondVec, newPixels, imageSize, newPixels.size, newLocations, dividingValue, channels) #now passes channels
                    segmentOne = segmentInfo['segOne']
                    segmentTwo = segmentInfo['segTwo']
                    posIndices = segmentInfo['posIndices']
                    negIndices = segmentInfo['negIndices']

                    # Calculate the weights of the edges that were removed from the image.
                    # Reduce the weight matrix to two new matrices, one for each image segment.
#                    edgeSum, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
                    cutSize, matrixOne, matrixTwo = weightMatrix.ReduceMatrix(posIndices, negIndices)
#                    matrixOneSum = numpy.sum(matrixOne.GetMatrix(), dtype=numpy.float64)
#                    matrixTwoSum = numpy.sum(matrixTwo.GetMatrix(), dtype=numpy.float64)
#                    cutSize = (edgeSum / matrixOneSum) + (edgeSum / matrixTwoSum)
#                    print "Size of cut = %f" % cutSize
                    if cutSize < prevCutSize:
#                        prevSum = edgeSum
                        prevCutSize = cutSize
                        prevMatrixOne = matrixOne
                        prevMatrixTwo = matrixTwo
                        prevSegInfo = segmentInfo

#                edgeSum = prevSum
                cutSize = prevCutSize
                matrixOne = prevMatrixOne
                matrixTwo = prevMatrixTwo
#                matrixOneSum = numpy.sum(matrixOne.GetMatrix(), dtype=numpy.float64)
#                matrixTwoSum = numpy.sum(matrixTwo.GetMatrix(), dtype=numpy.float64)
#                cutSize = (edgeSum / matrixOneSum) + (edgeSum / matrixTwoSum)
                print "Size of choosen cut = %f" % cutSize
                segmentInfo = prevSegInfo
                segmentOne = segmentInfo['segOne']
                segmentTwo = segmentInfo['segTwo']
                posIndices = segmentInfo['posIndices']
                negIndices = segmentInfo['negIndices']
                print "Pixels in image one = %d" % len(posIndices)
                print "Pixels in image two = %d" % len(negIndices)
                print "Matrix one size = %dx%d" % (matrixOne.GetMatrix().shape[0], matrixOne.GetMatrix().shape[1])
                print "Matrix two size = %dx%d" % (matrixTwo.GetMatrix().shape[0], matrixTwo.GetMatrix().shape[1])


            # Create two arrays of pixels from the original image using
            # the indices returned from DivideImage.
            posPixels = numpy.take(newPixels, posIndices)
            negPixels = numpy.take(newPixels, negIndices)

            # Save the pixel locations for each new pixel array.
            posLocations = numpy.take(newLocations, posIndices)
            negLocations = numpy.take(newLocations, negIndices)

#!            plotter.AddPlot(image, secondVec)

            filename = "segment_%d_%d" % (cutNumber, imageNumber)
            print "Writing image file %s.tif" % filename
            data.WriteNewImage(segmentOne, image_path + "/%s.tif" % filename)
            posArrays = {'pixels':posPixels, 'locations':posLocations}
            numpy.savez(pixel_path + "/%s.npz" % filename, **posArrays)
            numpy.save(matrix_path + "/%s.npy" % filename, matrixOne.GetMatrix())

            imageNumber = imageNumber + 1

            filename = "segment_%d_%d" % (cutNumber, imageNumber)
            print "Writing image file %s.tif\n" % filename
            data.WriteNewImage(segmentTwo, image_path + "/%s.tif" % filename)
            negArrays = {'pixels':negPixels, 'locations':negLocations}
            numpy.savez(pixel_path + "/%s.npz" % filename, **negArrays)
            numpy.save(matrix_path + "/%s.npy" % filename, matrixTwo.GetMatrix())
            imageNumber = imageNumber + 1

#!    plotter.ShowPlots()

    return

if __name__ == "__main__":
    maxPixelDistance = 8
    filename = "sampleblock.jpg"
   #filename = "/lena_32x32.tif"
   #image_dir = "/Users/roehrig/PycharmProjects/segment_image/images"
    image_dir = "C:\Users\joeho_000\OneDrive\Work\ANL APS XSD\images\\"
    data = ImageFileData("{}{}".format(image_dir, filename))

    #Set the type of dividing to be done.
    # 0 - divide the image using the value of zero
    # 1 - divide the image using the median value of the eignevector
    # 2 - try vector.size / (log(vector.size))^2 evenly spaced dividing points
    divideType = 2

    # Read in the image and initialize information such as image size,
    # mode (rgb, b&w, etc.)
    data.ReadImage()

    print("data: {}".format(list(data.GetImageData()))) #inspecting the output of getData
    print "Image mode is %s" % data.GetImageMode()
    imageData = data.GetImageData()
    imageSize = data.GetImageSize()
    dimensions = data.GetImageDimensions()
    channels = data.GetBands()                          #added bands/channels
    fileFormat = data.GetFileFormat()                   #added fileFormat
    print("Image format is " + fileFormat)
    # Create an array of pixel locations, location=sqrt(x^2 + y^2)
    locationValues = data.pixels.CreateLocationArray()
    print "Number of image pixels = %d" % imageSize
    print "Image width = %d, image height = %d" % dimensions
#    sigmaI = CalculateIntensitySigma(data.GetPixelsArray())
    sigmaI = numpy.var(imageData)
    print "Intensity variance = %f" % sigmaI
#    sigmaX = CalculateLocationSigma(data.GetPixels())
    sigmaX = numpy.var(locationValues)
    print "Location variance = %f" % sigmaX


    print "Creating weight matrix"
    weightMatrix = WM(data.size, data.size)
    weightMatrix.SetPixelData(data.GetPixels(), maxPixelDistance)
    weightMatrix.CreateMatrix(sigmaI, sigmaX)

    SegmentImage(weightMatrix, data, image_dir, divideType, channels, fileFormat) #now passes fileformat and channels
