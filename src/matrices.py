__author__ = 'roehrig'

import numpy
import pixel
import math
import multiprocessing as mp
import shareGui


def unwrap_CreateMatrix(args):                                                #neccessary function for process pool to work properly (target function of each child process cannot be inside of a class)
    return WeightMatrix.CreateMatrixPixelA(*args)


class Matrix():
    '''
    This represents a diagonal matrix, where only the values on the
    diagonal have a nonzero value.
    '''

    def __init__(self, columns, rows):
        '''
        columns - the number of columns in the matrix.
        rows - the number of rows in the matrix
        '''

        self.columns = columns
        self.rows = rows

        self.size = self.columns * self.rows
        self.data = numpy.zeros(self.size, numpy.float64)                      # creates an array populated by zeros with an item for every pixel, datatype float64

        self.matrix = None

        return

    def CreateMatrix(self):

        # Create the matrix.  Make sure that the data is in the correct shape first.
        self.matrix = numpy.matrix(self.data.reshape(self.columns, self.rows))

        return

    def GetMatrix(self):
        return self.matrix

    def SetMatrix(self, newMatrix):
        self.matrix = newMatrix
#        self.data = newMatrix.flatten()
        self.data = numpy.ravel(newMatrix)
        return

    def GetNumColumns(self):
        return self.columns

    def GetNumRows(self):
        return self.rows

    def GetMatrixSize(self):
        return self.size

    def SetMatrixData(self, data):
        self.data = data
        return

    # This function returns a flattened array
    def GetMatrixDataArray(self):
        return self.data

    def SaveToFile(self, path):
        numpy.save(path, self.matrix)
        return

class WeightMatrix(Matrix):
    '''
    This represents a weight matrix for a graph.  Each value is indicative
    of the weight of the edge that connects two vertices.  Vertices that
    are more tightly connected have a large edge weight than those that
    are less connected.
    '''
    def __init__(self, columns, rows):
        '''
        columns     - the number of columns in the matrix
        rows        - the number of rows in the matrix
        '''

        Matrix.__init__(self, columns, rows)
        return

    def SetMaxPixelDistance(self, newDistance=1):

        self.distance = newDistance
        return

    def SetPixelData(self, data, maxDistance=1):

        self.distance = maxDistance
        self.pixelArray = data.GetPixelArray()
        self.numPixels = data.GetPixelArraySize()

        return

    def CalcLocationVectorNorm(self, pixelA, pixelB):

        locA = numpy.array([pixelA.GetRowNumber(), pixelA.GetColumnNumber()], dtype=numpy.float64)
        locB = numpy.array([pixelB.GetRowNumber(), pixelB.GetColumnNumber()], dtype=numpy.float64)

        norm = numpy.linalg.norm((locA - locB), 2)

        return norm

    def CalcPixelScalarNorm(self, pixelA, pixelB):

        valueA = pixelA.GetValue()
        valueB = pixelB.GetValue()

        norm = math.sqrt((valueA - valueB) * (valueA - valueB))

        return norm

    def CalcPixelVectorNorm(self, pixelA, pixelB):

        try:                                                               #Will try to calulate the vector norm of the pixel values
            valueA = numpy.array(pixelA.GetValue(), dtype = numpy.float64) #But if pixel values are integers (grayscale), this will throw a ValueError
            valueB = numpy.array(pixelB.GetValue(), dtype = numpy.float64) #In that case, calculate a scalar norm insead
            norm = numpy.linalg.norm((valueA - valueB), 2)
            return norm

        except ValueError:
            return self.CalcPixelScalarNorm(pixelA, pixelB)


    def CreateMatrixPixelA(self, sigmaI=1, sigmaX=1, i = 0):               #rewritten to only calulate the relationships of one pixel at a time (numPixels # of loops instead of numPixels^2)
                                                                           #multiple processes now use this function
        pixelA = self.pixelArray[i]
        pixelAData = []

        j = 0
        for pixelB in self.pixelArray:
            stride = (i * self.numPixels) + j

            locationDiff = self.CalcLocationVectorNorm(pixelA, pixelB)

            if locationDiff < self.distance:
                intensityDiff = self.CalcPixelVectorNorm(pixelA, pixelB)   #Now calculates this inside of the if
                locationDiff = -1 * pow(locationDiff,2)
                intensityDiff = -1 * pow(intensityDiff, 2)                 #Changed the Intensity difference to calulate the vector norm of two arrays (works for multi-layer images)
                value = math.exp(intensityDiff / sigmaI) * math.exp(locationDiff / sigmaX)
                pixelAData.append((value, stride))

            j += 1

        return pixelAData                                                  #This is the data for one pixel. numPixels # of these returned to CreateMatrix to append all values to self.data


    def CreateMatrix(self, sigmaI, sigmaX):                                #Creates a process pool and distributes theyre work to all pixels, to calculate weight matrix

        gui = shareGui.getGui()
        cpus = mp.cpu_count()
        poolCount = cpus
        args = [(self, sigmaI, sigmaX, i,) for i in range(self.numPixels)]
        gui.updateLog('Number of cpu\'s to process WM:%d'%cpus)

        pool = mp.Pool(processes = poolCount)
        gui.updateLog('Mapping pool processes')
        tempData = pool.map(unwrap_CreateMatrix, args)

        for pixelList in tempData:                                          #This puts the data of each pixel, returned from each seperate process, into self.data
            for pixel in pixelList:
                self.data[pixel[1]] = pixel[0]

        self.matrix = numpy.matrix(self.data.reshape(self.columns, self.rows), numpy.float64)
        gui.updateLog('{}'.format(self.matrix.shape))
        return

    def ReduceMatrix(self, posIndices, negIndices):
        '''
        This function divides weight matrix into two parts, one for each segment
        that the image was divided into.  It also sums the weights of the edges
        that were removed.
        '''

        edgeSum = 0
        for i in range(self.rows):
            # Check to see if the row in the matrix is from a pixel in the
            # 'positive' side of the image.
            if i in posIndices:
                row = self.matrix[i]
                # Choose the weight from each row that correspond to pixels in the
                # 'negative' side of the image.
                edges = numpy.take(row, negIndices)
                edgeSum = edgeSum + edges.sum()

        # Select from the weight matrix only the rows that go with one
        # segment of the image.
        temp = numpy.take(self.matrix, posIndices, 0)
        # Select from the remaining rows only the columns that go
        # with the same segment of the image.
        temp2 = numpy.take(temp, posIndices, 1)
        # Create a new, smaller weight matrix.
        posMatrix = WeightMatrix(temp2.shape[0], temp2.shape[1])
        posMatrix.SetMatrix(temp2)

        sum1 = numpy.sum(temp, dtype=numpy.float64)

        # Select from the weight matrix only the rows that go with the
        # other segment of the image.
        temp = numpy.take(self.matrix, negIndices, 0)
        # Select from the remaining rows only the columns that go
        # with the other segment of the image.
        temp2 = numpy.take(temp, negIndices, 1)
        # Create a new, smaller weight matrix.
        negMatrix = WeightMatrix(temp2.shape[0], temp2.shape[1])
        negMatrix.SetMatrix(temp2)

        # Calculate cut size
        sum2 = numpy.sum(temp, dtype=numpy.float64)
        cutsize = (edgeSum / sum1) + (edgeSum / sum2)

#        return (edgeSum, posMatrix, negMatrix)
        return (cutsize, posMatrix, negMatrix)


class DiagonalMatrix(Matrix):
    '''
    This represents a diagonal matrix where each diagonal element is the
    sum of edge weights for all edges connected to a vertex.
    '''

    def __init__(self, columns, rows):

        Matrix.__init__(self, columns, rows)
        gui = shareGui.getGui()
        gui.updateLog("D matrix size=%d" % self.data.shape)

        return

    def CreateMatrix(self, weights):

        for i in range(self.size):
            self.data[i] = weights[i].sum()

        gui = shareGui.getGui()
        temp = numpy.diag(self.data)
        self.matrix = numpy.matrix(temp, numpy.float)
        gui.updateLog('{}'.format(self.matrix.shape))

        return
