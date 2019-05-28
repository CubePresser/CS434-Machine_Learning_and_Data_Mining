import numpy as np
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

#Adds a feature column to X with random values derived from a standard normal distribution
def addRandomFeatures(X):
    examples = len(X)
    samples = np.random.normal(0, 1, examples)
    for k in range(0, examples):
        X[k].append(samples[k])

    return np.array(X)

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

    return trASE, teASE

def main():
    #Get Args
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    d = 10
    trASEData = []
    teASEData = []

    #Get feature,result pairs from files
    trData, trRes = parseData(arg1)
    teData, teRes = parseData(arg2)

    #Get average squared errors and display resulting ASE's
    trASE, teASE = displayResults(trData, trRes, teData, teRes)

    #Add ASEs to the list of all ASEs for plot
    trASEData.append(trASE)
    teASEData.append(teASE)

    for i in range(1, d+1):

        #Add random features to both train and test data
        trData = addRandomFeatures(trData.tolist())
        teData = addRandomFeatures(teData.tolist())

        trASE, teASE = displayResults(trData, trRes, teData, teRes)

        trASEData.append(trASE)
        teASEData.append(teASE)

    #Create plot
    x = np.arange(0, d+1)
    plt.title('ASE as Random Features Increases')
    plt.xlabel('Number of Random Features')
    plt.ylabel('ASE')

    plt.plot(x, np.array(trASEData), 'b-', linewidth=2.0, label='Training ASE')
    plt.plot(x, np.array(teASEData), 'r-', linewidth=2.0, label='Testing ASE')
    plt.legend()
    plt.savefig('q1_4-PLOT')

main()
