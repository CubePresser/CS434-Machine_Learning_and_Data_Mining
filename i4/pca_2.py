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
    
    # Compute mean
    # Compute covariance matrix
    mean_vec = np.mean(data.values, axis=0)
    img_2D = []
    for i in range(28):
        offset = i * 28
        img_2D.append(mean_vec[offset:offset+28])
        
    img = Image.fromarray((np.array(img_2D)*256).astype(np.uint8))
    img.save('images/mean.png')

    #print('Mean \n%s' %mean_vec)
    cov_mat = (data.values - mean_vec).T.dot((data.values - mean_vec)) / (data.values.shape[0]-1)
    #print('Covariance matrix \n%s' %cov_mat)
    # Get top ten eigenvectors

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    #print('Eigenvectors \n%s' %eig_vecs)
    #print('\nEigenvalues \n%s' %eig_vals)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0])
    eig_pairs.reverse()

    top_ten = []
    for i in range(10):
        top_ten.append(eig_pairs[i][1])
    
    for k in range(10):
        img_2D = []
        for i in range(28):
            offset = i * 28
            img_2D.append(top_ten[k][offset:offset+28])
        
        img = Image.fromarray((np.array(img_2D)*256).astype(np.uint8))
        img.save('images/eigen_' + str(k) + '.png')
    
main()