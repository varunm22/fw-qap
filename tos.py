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

def tos(W, D, i_max = 1e4, X0=None, stop_tol = 0.0001):
    n = n_(W)

    # TODO: factor out some of this initialization logic
    if X0 == None:
        X = np.ones((n,n))/n
    elif X0 == -1: # random start near center of space
        flat = np.ones((n,n))/n
        lam = 0.5
        X = (1-lam)*flat + lam*sink(np.random.random(W.shape), 10)
    else: # actual permutation array provided
        X = X0

    # these can be tuned
    L = norm(W)*norm(D)
    L = L if L != 0 else 1e-6
    s = 1/L
    l = 1

    Y = X
    i = 0
    stop = 0
    while (i < i_max and stop == 0):
        X_old = np.copy(X)
        Z = proj1(Y)
        d = g_(Z, W, D)
        X = proj2(2*Z - Y - s*d)
        # print("Y", Y)
        # print("X_old", X_old)
        # print("Z", Z)
        # print("d", d)
        # print("X", X)
        Y += l*(X-Z)
        # print("Y", Y)
        # print(norm(X-X_old), norm(X))
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

def print_result(x):
    print("Final permutation: ", x[1])
    print("Final cost value: ", x[0])
    print("Final optimized x vector: ", x[2])
    print("Final number of iterations: ", x[3])
