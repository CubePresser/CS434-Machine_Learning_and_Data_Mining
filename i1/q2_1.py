import numpy as np
import math
import sys
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

maxPow = (math.log(sys.float_info.max))-0.1
minPow = (math.log(sys.float_info.min))+0.1

def readFile(fileName):
    f = open(fileName, 'r+')
    return csv.reader(f)

def parseCSV(path):
    data = readFile(path)
    grayscale = []
    digits = []
    for row in data:
        row = [float(item) for item in row]
        grayscale.append(row[0:256])
        digits.append(row[-1])
    grayscale = np.array(grayscale)
    digits = np.array(digits)

    return grayscale, digits

def accuracy(X, Y, w):
    success = 0
    n = len(X)
    for i in range(0, n):
        wT = np.multiply(w.T, -1)
        pow = np.dot(wT, X[i])
        if(pow >= maxPow):
            exp = 1
        elif(pow <= minPow):
            exp = 0
        else:
            exp = np.exp(pow)
        result = 1 / (1 + exp)

        if round(result, 1) == Y[i]:
            success = success + 1
    
    return ((float(success) / float(n)) * 100)

def main():
    itr = 50
    
    # trainingData = sys.argv[1]
    # testingData = sys.argv[2]
    learn = 0.01 #sys.argv[3]

    trGrayscale, trDigits = parseCSV('data/usps-4-9-train.csv')
    teGrayscale, teDigits = parseCSV('data/usps-4-9-test.csv')


    trainingAccuracy = []
    testingAccuracy = []

    for f in range(1, itr):
        n = len(trGrayscale)
        w = np.array([])
        for i in range(f):
            gradient = np.array([])
            for k in range(0, n):
                wT = np.multiply(w.T, -1)
                pow = np.dot(wT, trGrayscale[k]) if len(w) else np.multiply(-1, np.sum(trGrayscale[k]))
                if(pow >= maxPow):
                    exp = 1
                elif(pow <= minPow):
                    exp = 0
                else:
                    exp = np.exp(pow)
                result = 1 / (1 + exp)
                gradient = gradient + (trGrayscale[k] * np.subtract(result, trDigits[k])) if len(gradient) else (trGrayscale[k] * np.subtract(result, trDigits[k]))
            w = np.subtract(w, learn * gradient) if len(w) else learn*gradient
    
        trainingAccuracy.append(accuracy(trGrayscale, trDigits, w))
        testingAccuracy.append(accuracy(teGrayscale, teDigits, w))

    x = np.arange(1, itr)
    plt.xlabel('Gradient Descent Iterations')
    plt.ylabel('Accuracy')
    plt.title("Accuracy vs Descent Iterations")
    plt.plot(x, trainingAccuracy, 'b-', label="Training")
    plt.plot(x, testingAccuracy, 'r-', label="Testing")
    plt.legend()
    plt.savefig('q2_1-PLOT')

main()