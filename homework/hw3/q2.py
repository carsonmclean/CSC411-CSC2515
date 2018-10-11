# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp # scipy.misc throws warning
from sklearn.datasets import load_boston

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    # convert test vector (14,) to matrix (14,1)
    test_datum = test_datum.reshape(test_datum.shape[0], 1)

    a_i = np.exp(-l2(test_datum.T,x_train)/(2*(tau**2)))/np.exp(logsumexp(-l2(test_datum.T,x_train)/(2*(tau**2))))
    A = np.diagflat(a_i) # np.diag extracts diagonal, np.diagflat makes diag matrix from input
    # w = (XT*A*X + lambda*I)^-1 XT*A*y
    w = np.linalg.solve(x_train.T.dot(A).dot(x_train) + lam * np.identity(x_train.shape[1]), x_train.T.dot(A).dot(y_train))
    prediction = test_datum.T.dot(w)
    return prediction


def run_validation(x, y, taus, val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    # shuffle dataset and split into train and validate
    # https://play.pixelblaster.ro/blog/2017/01/20/how-to-shuffle-two-arrays-to-the-same-order/
    shuffle_order = np.arange(x.shape[0])
    x_shf = x[shuffle_order]
    y_shf = y[shuffle_order]
    cutoff = int(len(x) * (1 - val_frac))
    x_tra, x_val = x_shf[:cutoff], x_shf[cutoff:]
    y_tra, y_val = y_shf[:cutoff], y_shf[cutoff:]

    tau = 1.0
    lrls = LRLS(x_tra[22], x_tra, y_tra, tau)
    print(lrls)

    return None, None


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)
