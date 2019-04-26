import numpy as np
import csv
import sys
import random as rand
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# class representing one node of the tree
class Node(object):
    def __init__(self):
        self.left = 0
        self.right = 0
        self.label = 0
        self.featureIndex = 0
        self.split = 0

# read file
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

# parse the CSV
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

# calculate info gain
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

# split the set
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

# sort the examples in ascending order of column i
def sort(examples, i):
    return np.array(sorted(examples, key=lambda x: x[i]))

# print the given node
def printNode(node):
    print ('Label: ' + str(node.label))
    print('Feature: ' + str(node.featureIndex))
    print('Split: ' + str(node.split))

# Follow path through decision tree and return error for one example
def followError(tree, example, result):
    feat = tree.featureIndex
    split = tree.split
    
    # Left or right?
    if example[feat] <= split:
        node = tree.left
        # Has children else calc error
        if node.left and node.right:
            return followError(node, example, result)
        else:
            pos = node.label[0] > node.label[1]

            if (pos and (result < 0)):
                return 1
            else:
                return 0
    else:
        node = tree.right
        # Has children else calc error
        if node.left and node.right:
            return followError(node, example, result)
        else:
            pos = node.label[0] > node.label[1]

            if (pos and (result < 0)):
                return 1
            else:
                return 0

def getError(set, tree):
    error = 0
    for i in range(len(set)):
        error += followError(tree, set[i], set[i][0])

    print error
    return error

def createTree(set, memo, d):
    if(d == 0):
        return 0
    l1 = getCountedLabel(set.T[0])
    u1 = evalUncertainty(l1)

    bestFeature = (0, 0)
    bestGain = 0
    greatestLeft = 0
    greatestRight = 0

    stump = Node()
    stump.label = l1

    for i in range(1, len(set[0])):
        
        # check if already visited feature
        hasSeen = False
        for k in range(len(memo)):
            if i is memo[k]:
                hasSeen = True
                break
        if hasSeen:
            continue

        sortedExamples = sort(set, i)
        midpoints = []
        maxGain = 0
        bestSplit = 0
        bestLeft = 0
        bestRight = 0
    
        for a in range(2, len(sortedExamples)):
            midpoints.append(sortedExamples[a][i] - sortedExamples[a-1][i])
            
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
                bestLeft = left
                bestRight = right
            # add to appropriate child
        
        
        if maxGain > bestGain:
            bestGain = maxGain
            bestFeature = (i, bestSplit)
            stump.featureIndex = i
            stump.split = bestSplit
            greatestRight = bestRight
            greatestLeft = bestLeft
        
        # calculate info gain
        # if info gain is larger than max
        #   store state of current tree    
    
    # HAS STUMP NOW
    memo.append(bestFeature[0])
    newLeft = 0
    newRight = 0
    if(stump.left.label[0] and stump.left.label[1]):
        newLeft = createTree(np.array(greatestLeft), memo, d - 1)
    if newLeft:
        stump.left = newLeft

    if(stump.right.label[0] and stump.right.label[1]):
        newRight = createTree(np.array(greatestRight), memo, d - 1)
    if newRight:
        stump.right = newRight

    return stump

def main():
    Xtrain, Ytrain = parseCSV(sys.argv[1])
    Xtest, Ytest = parseCSV(sys.argv[2])

    dValues = []
    d = 1

    for i in range(len(Ytrain)):
        Xtrain[i][0] = Ytrain[i]
        
    for i in range(len(Ytest)):
        Xtest[i][0] = Ytest[i]

    tree = createTree(Xtrain, [], 4)
    printNode(tree)

    trainError = float(getError(Xtrain, tree)) / float(len(Xtrain))
    #testError = float(getError(Xtest, tree)) / float(len(Xtest))

    print 'Training Error: ' + str(trainError)
    print 'Testing Error: ' + str(testError)
    
   
    # create plot and table    
    print("creating plot")
    x = np.arange(1, d)
    plt.xlabel('d')
    plt.ylabel('Error')
    plt.title("Error as d increases")
    plt.plot(x, trainingErrors, 'b-', label="Training Error")
    plt.plot(x, testingErrors, 'r-', label="Testing Error")
    plt.legend(loc='best')
    
    columns = ['Training Error', 'Testing Error']
    rows = x
    tabledata = zip(trainingErrors, testingErrors)
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=tabledata,
                      rowLabels=rows,
                      colLabels=columns,
                      loc='bottom')
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig('errorvd-PLOT')
    
    

main()