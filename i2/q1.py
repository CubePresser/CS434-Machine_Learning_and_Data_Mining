import numpy as np
import csv
import sys
import random as rand
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# read from file
def readFile(fileName):
    f = open(fileName, 'r+')
    return csv.reader(f)    

# normalize the data
def normalizeData(X):
    X = X.T
    for col in X:
        min = col[0]
        max = col[0]
        for val in col:
            if(val < min):
                min = val
            elif(val > max):
                max = val       
        for i in range(0, len(col)):
            val = col[i]
            col[i] = (val - min) / (max - min)
            if (col[i] < 0) or (1 < col[i]):
                print("ERROR: ", col[i])

    return X.T

# parse the csv file to retrieve data
def parseCSV(path):
    data = readFile(path)
    Y = []
    X = []
    for row in data:
        row = [float(item) for item in row]
        Y.append(row[0])
        X.append(row[1:(len(row)- 1)])

    X = normalizeData(np.array(X))

    return X, Y

# break the data into folds
def getTrainTestFold(S, i):
    
    test = [S[i]]
    trains = []

    for a in range(len(S)):
        if a != i:
            trains.append(S[a])     
    return np.array(trains), np.array(test)

# calculate the distance between two vectors
def distance(vec1, vec2):
    result = np.subtract(vec1, vec2)
    mag = np.dot(result.T, result)
    dist = math.sqrt(mag)
    return dist

# get distance of every point to the target and sort by closest
def getDistances(t, X, Y):
    distances = []

    for i in range(len(X)):
        dist = distance(t, X[i])
        distances.append((dist, Y[i]))

    distances.sort()
    return distances

# tally votes from nearby neighors
def getError(result, distances, k):
    votes = 0
    for i in range(k):
        votes += (distances[i])[1]
    if (votes * result) < 0: 
        return 1
    return 0

# calculate error for leave out one cross validation
def loocvCalculation(set, results, k):
    error = 0
    numExamples = len(set)
    for i in range(numExamples):
        distances = getDistances(set[i], set, results)
        error += getError(results[i], distances[1:], k)
    return error

# calculate error
def calculateSetError(trSet, teSet, trRes, teRes, k):
    error = 0
    numExamples = len(teSet)
    for i in range(numExamples):
        distances = getDistances(teSet[i], trSet, trRes)
        error += getError(teRes[i], distances, k)
    return error

def main():
    Xtrain, Ytrain = parseCSV(sys.argv[1])
    Xtest, Ytest = parseCSV(sys.argv[2])

    S = Xtrain
    R = Ytrain

    numExamples = len(S[0])
    trainingErrors = []
    testingErrors = []
    loocvs = []

    # for each value of k
    for k in range(1, numExamples, 2):

        # calculate error
        print "K VALUE: " + str(k) + "================================="
        trainingError = calculateSetError(Xtrain, Xtrain, Ytrain, Ytrain, k)
        testingError  = calculateSetError(Xtrain, Xtest, Ytrain, Ytest, k)
        loocv = loocvCalculation(Xtrain, Ytrain, k)
        trainingErrors.append(trainingError)
        testingErrors.append(testingError)
        loocvs.append(loocv)

        print "Training Error: " + str(float(trainingError)/float(len(Xtrain)))
        print "LOOCV Error: " + str(float(loocv)/float(len(Xtrain)))
        print "Testing Error: " + str(float(testingError)/float(len(Xtest)))

    #create plot
    print("creating plot")
    x = np.arange(1, numExamples, 2)
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.title("Error as k increases")
    plt.plot(x, trainingErrors, 'b-', label="Training Error")
    plt.plot(x, testingErrors, 'r-', label="Testing Error")
    plt.plot(x, loocvs, 'g-', label="Cross Validation Error")
    plt.legend(loc='best')
    plt.savefig('errorvk-PLOT')
main()