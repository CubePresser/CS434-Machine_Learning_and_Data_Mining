import numpy as np
import pandas as pd
import sys
import matplotlib as mpl

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# read from file
def readFile(fileName):
    f = open(fileName, 'r+')
    return csv.reader(f)    

# parse the csv file to retrieve data
def parseCSV(path):
    data = readFile(path)
    X = []
    for row in data:
        row = [float(item) for item in row]
        X.append(row)

    return X

# Calculate euclidean distance between two vectors
def dist(v1, v2):
    return np.linalg.norm(v1 - v2)

def cluster(data, centroids):
    clusters = []
    for i in range(len(centroids)):
        clusters.append([])
    # For each example in the set
    for i in range(len(data)):
        example = data.iloc[i]
        minDist = sys.maxsize
        closest = 0
        # For each of the centroids get index of centroid closest to current example
        for k in range(len(centroids)):
            centroid = centroids.iloc[k].values
            distance = dist(example, centroid)
            # If calculated distance is shorter than previous minimum
            if(distance < minDist):
                minDist = distance
                closest = k

        # Add example index to appropriate cluster
        clusters[closest].append(i)
    
    return clusters


def getCenter(data, cluster):
    sumV = np.zeros(784)
    # Get sum vector
    for i in range(len(cluster)):
        sumV = np.add(sumV, data.iloc[cluster[i]].values)
    
    centroid = np.true_divide(sumV, len(cluster))
    centroidFrame = pd.Series(centroid)

    return centroidFrame

def adjustCentroids(data, centroids, clusters):
    for i in range(len(centroids)):
        centroids.iloc[i] = getCenter(data, clusters[i])
    
    return centroids

def SSE(data, centroids, clusters):
    sse = 0
    # For each cluster
    for i in range(len(clusters)):
        cluster = clusters[i]
        # For every example in that cluster
        for c in range(len(cluster)):
            sse += dist(data.iloc[cluster[c]].values, centroids.iloc[i].values) ** 2
    
    return sse

def main():
    data = pd.read_csv('p4-data.txt', header=None).divide(255)
    iterations = 10
    kSSEs = [0] * 9

    for k in range(2, 11):
        lowestSSE = sys.maxsize
        for iter in range(10):
            # pick K random points for centroids
            C = data.sample(k)
            
            for i in range(iterations):
                print('k = ', k, ', Big Iteration ', str(iter), ', Iteration ', i)
                clusters = cluster(data, C)
                C = adjustCentroids(data, C, clusters)
                sse = SSE(data, C, clusters)
                print('SSE: ', sse)
                if sse < lowestSSE:
                    lowestSSE = sse
        print('lowestSSE for k=',k,lowestSSE)
        kSSEs[k-2] = lowestSSE

    # plotting
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Error')
    plt.plot(np.arange(2, 11), kSSEs, 'b-')
    plt.title('SSE vs k')
    plt.savefig('SSEvK')
main()