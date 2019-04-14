import numpy as np
import sys

def parseData(path):
    X = []
    Y = []
    with open(path) as file:
        for line in file:
            row = [float(item) for item in line.split()]
            X.append(row[0:13])
            Y.append(row[13])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def calcWeight(X, Y):
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)

def calcResults(X, W):
    results = []
    for row in X:
        results.append(np.dot(row, W))
    return results

def getASE(Y0, Y1):
    n = len(Y0)
    sum = 0
    for i in range(1, n):
        sum += (Y0[i] - Y1[i])**2
    return (sum / n)

def displayResults(trData, trRes, teData, teRes):
    weight = calcWeight(trData, trRes)

    print("Learned Weight Vector")
    print(weight)

    trY1 = calcResults(trData, weight)
    teY1 = calcResults(teData, weight)
    trASE = getASE(trRes, trY1)
    teASE = getASE(teRes, teY1)

    print("Training ASE")
    print(trASE)

    print("Testing ASE")
    print(teASE)

def main():
    #Get Args
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    #Get feature,result pairs from files
    trData, trRes = parseData(arg1)
    teData, teRes = parseData(arg2)

    displayResults(trData, trRes, teData, teRes)

main()
