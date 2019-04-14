import numpy as np
import sys
import csv

def readFile(fileName):
    f = open(fileName, 'r+')
    return csv.reader(f)

def main():
    data = readFile("data/usps-4-9-train.csv")
    trainGrayscale = []
    trainDigits = []
    for row in data:
        trainGrayscale.append(row[0:256])
        trainDigits.append(row[256: 257])
    trainGrayscale = np.double(trainGrayscale)
    trainDigits = np.double(trainDigits)

    data = readFile("data/usps-4-9-test.csv")
    testGrayscale = []
    testDigits = []
    for row in data:
        testGrayscale.append(row[0:256])
        testDigits.append(row[256: 257])
    testGrayscale = np.double(testGrayscale)
    testDigits = np.double(testDigits)
main()