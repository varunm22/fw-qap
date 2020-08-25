import math
from utils import *
from line_search import *
import numpy as np

# TODO: get this to work for simple tests
# TODO: figure out why x has 2n^2 elements instead of n^2

def flat_concat(x1,x2):
    x1_flat = x1.reshape(x1.size)
    x2_flat = x2.reshape(x2.size)
    return np.concatenate((x1_flat, x2_flat))

def sfw(A, B, i_max = 30, x0=None, stop_tol = 0.0001):
    n = n_(A)
    stype = 2 # some parameter for line search, this assumes function is quadratic
    if x0 == None:
        # if i_max == 0.5, use identity as starting point and perform
        # one iteration of FW with step length 1
        if i_max == 0.5:
            t = np.eye(n)
            x0 = flat_concat(t,t)
        else:
            x0 = np.concatenate((1/n*np.ones(n*n), 1/n*np.ones(n*n)))
    elif x0 == -1: # random start near center of space
        X = np.ones(n)/n
        lam = 0.5
        X = (1-lam)*X + lam*sink(np.random.random(A.shape), 10)
        Y = X
        x0 = flat_concat(X, Y)
    elif x0.shape == (A.shape[0],): # permutation provided as 1d array
        x0 = perm2mat(x0).T
        x0 = flat_concat(x0,x0)
    elif x0.shape == A.shape:
        x0 = flat_concat(x0,x0)
    else:
        pass

    x = x0
    myp = []
    i = 0
    stop = 0
    myps = np.zeros((math.ceil(i_max), n))
    while (i < i_max and stop == 0):
        f0, g = fungrad(x, A, B)
        g_ = (g[:n*n]+g[n*n:])/2
        g = np.concatenate((g_,g_))
        d, myp = dsproj(x,g,n)
        stop_norm = d.dot(d)
        if stop_norm < stop_tol:
            stop = 1
        if i_max > 0.5:
            f0_new, salpha = line_search(stype, x, d, g, A, B)
        else:
            salpha = 1
        x = x + salpha*d
        i += 1
        if salpha == 0:
            stop = 1
        # TODO: add the list output computations here

    if salpha != 1:
        P, Q = unstack (x, n)
        myp, _, _ = assign(P,True)

    P = perm2mat(myp)
    f = np.sum(P.dot(A).dot(P.T)*B)/2
    return (f, myp, x, i)

def print_result(x):
    print("Final permutation: ", x[1])
    print("Final cost value: ", x[0])
    print("Final optimized x vector: ", x[2])
    print("Final number of iterations: ", x[3])
