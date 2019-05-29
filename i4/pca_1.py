import numpy as np
import pandas as pd
import sys

def main():
    data = pd.read_csv('p4-data.txt', header=None).divide(255)

    # Eigen calculation code obtained from:
    # https://plot.ly/ipython-notebooks/principal-component-analysis/#1--eigendecomposition--computing-eigenvectors-and-eigenvalues
    
    # Compute mean
    # Compute covariance matrix
    mean_vec = np.mean(data.values, axis=0)
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
        top_ten.append(eig_pairs[i][0])
    
    print('Top Ten eigenvalues')
    for i in top_ten:
        print(i)
    

    
main()