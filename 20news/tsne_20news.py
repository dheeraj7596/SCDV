#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as Math
import pylab as Plot
import pandas as pd
import sys
import matplotlib.cm as cm
import matplotlib as mpl


def Hbeta(D=Math.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = Math.exp(-D.copy() * beta)
    sumP = sum(P)
    H = Math.log(sumP) + beta * Math.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=Math.array([]), tol=1e-5, perplexity=30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = Math.sum(Math.square(X), 1)
    D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X)
    P = Math.zeros((n, n))
    beta = Math.ones((n, 1))
    logU = Math.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -Math.inf
        betamax = Math.inf
        Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while Math.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == Math.inf or betamax == -Math.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == Math.inf or betamin == -Math.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # Set the final row of P
        P[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta)))
    return P


def pca(X=Math.array([]), no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - Math.tile(Math.mean(X, 0), (n, 1))
    (l, M) = Math.linalg.eig(Math.dot(X.T, X))
    Y = Math.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=Math.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1500
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = Math.random.randn(n, no_dims)
    dY = Math.zeros((n, no_dims))
    iY = Math.zeros((n, no_dims))
    gains = Math.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + Math.transpose(P)
    P = P / Math.sum(P)
    P = P * 4  # early exaggeration
    P = Math.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = Math.sum(Math.square(Y), 1)
        num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0
        Q = num / Math.sum(num)
        Q = Math.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = Math.sum(Math.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - Math.tile(Math.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = Math.sum(P * Math.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    # Return solution
    return Y


if __name__ == "__main__":
    label_train = pd.read_csv('data/train_v2.tsv', header=0, delimiter="\t")
    label_test = pd.read_csv('data/test_v2.tsv', header=0, delimiter="\t")
    labels = Math.concatenate((label_train["class"], label_test["class"]), axis=0)
    gwbowv_name = "SDV_" + sys.argv[2] + "cluster_" + sys.argv[1] + "feature_matrix_gmm_sparse.npy"
    test_gwbowv_name = "TEST_SDV_" + sys.argv[2] + "cluster_" + sys.argv[1] + "feature_matrix_gmm_sparse.npy"
    train = Math.load(gwbowv_name)
    test = Math.load(test_gwbowv_name)
    X = Math.concatenate((train, test), axis=0)
    print("Running t-SNE on 20 news groups SDV vectors")
    cmap = cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    N = 20
    # define the bins and normalize
    bounds = Math.linspace(0, N, N + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    Y = tsne(train, 2, 50, 20.0)
    Plot.scatter(Y[:, 0], Y[:, 1], 20, c=label_train["class"], cmap=cmap, norm=norm)
    Plot.show()
