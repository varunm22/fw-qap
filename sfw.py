import math
from utils import *
from line_search import *
import numpy as np

# TODO: refactor all for square matrices
# TODO: figure out why x has 2n^2 elements instead of n^2

def flat_concat(x1,x2):
    x1_flat = x1.reshape(x1.size)
    x2_flat = x2.reshape(x2.size)
    return np.concatenate((x1_flat, x2_flat))

def sfw(A, B, i_max = 30, x0=None, stop_tol = 0.0001):
    m, n = A.shape
    stype = 2 # some parameter for line search, this assumes function is quadratic
    if x0 == None:
        # if i_max == 0.5, use identity as starting point and perform
        # one iteration of FW with step length 1
        if i_max == 0.5:
            t = np.eye(m)
            x0 = flat_concat(t,t)
        else:
            x0 = np.concatenate((1/m*np.ones(m*m), 1/n*np.ones(n*n)))
    elif x0 == -1: # random start near center of space
        X = np.ones(m)/m
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
    iter = 0
    stop = 0
    myps = np.zeros((math.ceil(i_max), n))
    while (iter < i_max and stop == 0):
        f0, g = fungrad(x, A, B)
        g_ = (g[:n*n]+g[n*n:])/2
        g = np.concatenate((g_,g_))
        d, myp = dsproj(x,g,m,n)
        stop_norm = d.dot(d)
        if stop_norm < stop_tol:
            stop = 1
        if i_max > 0.5:
            f0_new, salpha = line_search(stype, x, d, g, A, B)
        else:
            salpha = 1
        x = x + salpha*d
        iter += 1
        if salpha == 0:
            stop = 1
        # TODO: add the list output computations here

    if salpha != 1:
        P, Q = unstack (x, m, n)
        myp, _, _ = assign(P,True)

    f = np.sum(A*(B[myp,:][:,myp]))
    return (f, myp, x, iter)

def print_result(x):
    print(f"Final permutation: {x[1]}\n")
    print(f"Final cost value: {x[0]}\n")
    print(f"Final optimized x vector: {x[2]}\n")
    print(f"Final number of iterations: {x[3]}\n")

# sample problem:
# A is flow, B is distance
# 0 3 1      0 1 2
# 3 0 2      1 0 3
# 1 2 0      2 3 0

A = np.array([[0,3,1],[3,0,2],[1,2,0]])
B = np.array([[0,1,2],[1,0,3],[2,3,0]])
i_max = 1
print_result(sfw(A,B,i_max))
