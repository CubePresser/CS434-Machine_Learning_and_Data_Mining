from sklearn.cluster import KMeans
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
    k = int(sys.argv[1])
    
    # pick K random points for centroids
    C = data.sample(k)

    sseByIteration = [0] * iterations
    for i in range(iterations):
        print('Iteration ', i)
        clusters = cluster(data, C)
        C = adjustCentroids(data, C, clusters)
        sseByIteration[i] = SSE(data, C, clusters)
        print('SSE: ', sseByIteration[i])

    # plotting
    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Sum of Squared Error')
    plt.plot(np.arange(1, iterations + 1), sseByIteration, 'b-')
    plt.title('SSE for k=' + str(k))
    plt.savefig('SSE_k' + str(k))
    
main()