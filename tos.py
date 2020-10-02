import math
from utils import *
import numpy as np
from numpy.linalg import norm

def proj1(X):
    return X.clip(0)

def proj2(X):
    n = X.shape[0]
    one = np.ones((n,1))
    Z = (1/n + np.sum(X)/n**2)*np.eye(n) - (1/n)*X
    return X + Z.dot(one).dot(one.T) - 1/n*one.dot(one.T).dot(X)

def tos(W, D, X0, i_max = 1e4, stop_tol = 0.0001):
    n = n_(W)
    # these can be tuned
    L = norm(W)*norm(D)
    L = L if L != 0 else 1e-6
    s = 1/L
    l = 1

    X = X0
    Y = X
    i = 0
    stop = 0
    while (i < i_max and stop == 0):
        X_old = np.copy(X)
        Z = proj1(Y)
        d = g_(Z, W, D)
        X = proj2(2*Z - Y - s*d)
        Y += l*(X-Z)
        if (norm(X-X_old)/max(1,norm(X))) < stop_tol:
            stop = 1

        i += 1

    # if i == i_max:
    #     print("did not achieve convergence")

    _, p_out, _ = lap.lapjv(-X)
    # TODO: why does this need the T, but the sfw part at this step doesn't
    P = perm2mat(p_out).T
    f = f_(P, W, D)
    return (f, p_out, X, i)
