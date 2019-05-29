import numpy as np
import pandas as pd
import sys
import matplotlib as mpl
from PIL import Image

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


def main():
    data = pd.read_csv('p4-data.txt', header=None).divide(255)

    # Eigen calculation code obtained from:
    # https://plot.ly/ipython-notebooks/principal-component-analysis/#1--eigendecomposition--computing-eigenvectors-and-eigenvalues
    
    mean_vec = np.mean(data.values, axis=0)

    cov_mat = (data.values - mean_vec).T.dot((data.values - mean_vec)) / (data.values.shape[0]-1)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0])
    eig_pairs.reverse()

    top_ten = []
    for i in range(10):
        top_ten.append(np.array(eig_pairs[i][1]))

    matrix_w = np.hstack((top_ten[0].reshape(784,1), top_ten[1].reshape(784,1)))
    for i in range(8):
        matrix_w = np.hstack((matrix_w, top_ten[i+2].reshape(784,1)))
    
    reducedData = data.values.dot(matrix_w)
    
    maxIndices = [0] * 10
    for i in range(6000):
        for k in range(10):
            if reducedData[i][k] > reducedData[maxIndices[k]][k]:
                maxIndices[k] = i

    for k in range(10):
        img_2D = []
        for i in range(28):
            offset = i * 28
            img_2D.append(data.values[maxIndices[k]][offset:offset+28])
        
        img = Image.fromarray((np.array(img_2D)*256).astype(np.uint8))
        img.save('images/10Dimension_' + str(k) + '.png')
    
    
main()