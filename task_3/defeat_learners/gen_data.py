"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    # let number of rows in X be 5000 and columns be 10
    X = np.random.normal(size=(5000, 10))
    # taking sample of 1000 rows of X
    X = X[np.random.randint(X.shape[0], size=1000), :]
    Y = np.zeros(1000)
    for i  in range(0,10):
        Y += 3*X[:,i]
    return X, Y

# this function should return a dataset (X and Y) that will work
# better for decision trees than linear regression
def best4DT(seed=1):
    np.random.seed(seed)
    # let number of rows in X be 5000 and columns be 10
    X = np.random.normal(size=(5000, 10))
    # taking sample of 1000 rows of X
    X = X[np.random.randint(X.shape[0], size=1000), :]
    Y = np.zeros(1000)
    for col in range(0,10):
        Y += X[:, col] ** 3
    return X, Y

def author():
    return 'rajat & krunal' #Change this to your user ID


if __name__=="__main__":
    print ("they call me tikko.")
