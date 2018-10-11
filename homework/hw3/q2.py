# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp # scipy.misc throws warning
from sklearn.datasets import load_boston
import sys
import time

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

    A = np.diagflat(np.exp(-l2(test_datum.T, x_train)/(2*(tau**2)))/np.exp(logsumexp(-l2(test_datum.T, x_train)/(2*(tau**2)))))
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
    x_shf = x[idx]
    y_shf = y[idx]

    cutoff = int(len(x) * (1 - val_frac))
    x_tra, x_val = x_shf[:cutoff], x_shf[cutoff:]
    y_tra, y_val = y_shf[:cutoff], y_shf[cutoff:]

    tra_loss = []
    val_loss = []

    start = time.time()
    for i, tau in enumerate(taus):
        # Progress Bar with % Complete
        # Steven C. Howell
        # https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage/29703127#comment80545022_3002100
        sys.stdout.write('\r')
        now = time.time()
        elapsed = now - start
        k = (i + 1) / taus.shape[0]
        sys.stdout.write("[%-40s] %d%%    %d seconds remaining" % ('='*int(40*k), 100*k, elapsed/k - elapsed))
        sys.stdout.flush()

        tra_loss_i = 0.0
        val_loss_i = 0.0
        for j, _ in enumerate(x_tra):
            lrls = LRLS(x_tra[j], x_tra, y_tra, tau)
            tra_loss_i += (lrls - y_tra[j])**2
        for j, _ in enumerate(x_val):
            lrls = LRLS(x_val[j], x_tra, y_tra, tau)
            val_loss_i += (lrls - y_val[j])**2
        tra_loss.append(tra_loss_i)
        val_loss.append(val_loss_i)

    return tra_loss, val_loss


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(train_losses, label="train")
    plt.semilogx(test_losses, label="test")
    plt.legend()
    plt.show()
