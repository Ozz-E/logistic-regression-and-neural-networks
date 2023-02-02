import numpy as np
from sigmoid import *

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    hypothesis = 0.0

    #hypothesis = np.dot(X[i], theta)
    hypothesis = np.matmul(X[i], theta)
    
    result = sigmoid(hypothesis)

    return result
