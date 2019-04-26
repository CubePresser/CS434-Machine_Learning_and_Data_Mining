import numpy as np
import csv
import sys
import random as rand
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self):
        self.left = 0
        self.right = 0
        self.label = 0
        self.featureIndex = 0
        self.split = 0

def readFile(fileName):
    f = open(fileName, 'r+')
    return csv.reader(f)    

# normalize
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

# parse CSV input into matrices
def parseCSV(path):
    data = readFile(path)
    Y = []
    X = []
    for row in data:
        row = [float(item) for item in row]
        Y.append(row[0])
        X.append(row)

    X = normalizeData(np.array(X))

    return X, Y

# calculate the info gain
def evalInfoGain(u1, u2, u3, l1, l2, l3):
    t1 = l1[0] + l1[1]
    t2 = l2[0] + l2[1]
    t3 = l3[0] + l3[1]

    p1 = float(t2) / float(t1)
    p2 = float(t3) / float(t1)

    return u1 - ((p1*u2) + (p2*u3))

# calculate uncertainty
def evalUncertainty(label):
    total = float(label[0] + label[1])
    pR = float(label[0]) / total
    nR = float(label[1]) / total

    pU = 0
    nU = 0
    if pR:
        pU = pR * math.log(pR, 2)
    if nR:
        nU = nR * math.log(nR, 2)

    return -pU - nU

# calculate the split
def splitSet(examples, index, spl):
    left = []
    right = []

    for i in range(len(examples)):
        if examples[i][index] <= spl:
            left.append(examples[i])
        else:
            right.append(examples[i])

    return np.array(left), np.array(right)

# Returns number of positive and negative values in a pair
def getCountedLabel(Y):
    positive = 0
    negative = 0
    for i in range(len(Y)):
        if Y[i] > 0:
            positive += 1
        else:
            negative += 1
    
    return (positive, negative)

# sort the passed in matrix based off the column at index i
def sort(examples, i):
    return np.array(sorted(examples, key=lambda x: x[i]))

# print the given node
def printNode(node):
    print ('Label: ' + str(node.label))
    print('Feature: ' + str(node.featureIndex))
    print('Split: ' + str(node.split))
    
# calculate error
def getError(set, stump):
    error = 0
    feat = stump.featureIndex
    split = stump.split

    lpos = stump.left.label[0] > stump.left.label[1]
    rpos = not lpos

    for i in range(len(set)):
        if set[i][feat] <= split:
            if lpos and (set[i][0] < 0):
                error += 1
            elif rpos and (set[i][0] > 0):
                error += 1
        else:
            if rpos and (set[i][0] < 0):
                error += 1
            elif lpos and (set[i][0] > 0):
                error += 1
    
    return error

def main():
    Xtrain, Ytrain = parseCSV(sys.argv[1])
    Xtest, Ytest = parseCSV(sys.argv[2])

    for i in range(len(Ytrain)):
        Xtrain[i][0] = Ytrain[i]
        
    for i in range(len(Ytest)):
        Xtest[i][0] = Ytest[i]

    sortedExamples = Xtrain
    
    l1 = getCountedLabel(Ytrain)
    u1 = evalUncertainty(l1)

    bestFeature = (0, 0)
    bestGain = 0

    stump = Node()
    stump.label = l1

    # for every feature
    for i in range(1, len(Xtrain[0])):
        sortedExamples = sort(sortedExamples, i)
        midpoints = []
        maxGain = 0
        bestSplit = 0
    
        # for every example value of a specific feature calculate and store the midpoints
        for a in range(2, len(sortedExamples)):
            midpoints.append(sortedExamples[a][i] - sortedExamples[a-1][i])

        # find max gain for a specific feature using the midpoints
        for midpoint in midpoints:
            left, right = splitSet(sortedExamples, i, midpoint)
            l2 = getCountedLabel(left.T[0])
            l3 = getCountedLabel(right.T[0])

            u2 = evalUncertainty(l2)
            u3 = evalUncertainty(l3)

            infoGain = evalInfoGain(u1, u2, u3, l1, l2, l3)

            if infoGain > maxGain:
                maxGain = infoGain
                bestSplit = midpoint
                stump.left = Node()
                stump.left.label = l2
                stump.right = Node()
                stump.right.label = l3
        
        # find the max gain of all the features
        if maxGain > bestGain:
            bestGain = maxGain
            bestFeature = (i, bestSplit)
            stump.featureIndex = i
            stump.split = bestSplit
        

    #print required outputs
    print('\nRoot Node')
    printNode(stump)

    print('\nLeft Child')
    printNode(stump.left)
    
    print('\nRight Child')
    printNode(stump.right)

    print '\nInformation Gain: ' + str(bestGain)

    trainError = float(getError(Xtrain, stump)) / float(len(Xtrain))
    testError = float(getError(Xtest, stump)) / float(len(Xtest))

    print 'Training Error: ' + str(trainError)
    print 'Testing Error: ' + str(testError)

main()