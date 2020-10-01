import math
from utils import *
import numpy as np

def sfw(W, D, i_max = 30, x0=None, stop_tol = 0.0001):
    # TODO: factor out this initialization logic
    n = n_(W)
    if x0 == None:
        x0 = np.ones((n,n))/n
    elif x0 == -1: # random start near center of space
        X = np.ones((n,n))/n
        lam = 0.5
        x0 = (1-lam)*X + lam*sink(np.random.random(W.shape), 10)
    elif x0.shape == (W.shape[0],): # permutation provided as 1d array
        x0 = perm2mat(x0).T
    else: # actual permutation array provided
        pass

    X = x0
    myp = []
    i = 0
    stop = 0
    myps = np.zeros((math.ceil(i_max), n))
    while (i < i_max and stop == 0):
        f0, g = f_(X, W, D), g_(X, W, D)
        _, q, _ = lap.lapjv(g.T)
        Q = perm2mat(q)
        d = Q-X
        gap = np.sum(g*Q)
        if gap/f0 < stop_tol:
            stop = 1

        # try other step sizes! 
        # TODO: try alp's step size formula
        # eta = 2/(i+1)
        _, eta = line_search(X, d, g, W, D)

        X = X + eta*d
        i += 1

    _, p_out, _ = lap.lapjv(-X)
    P = perm2mat(p_out)
    f = f_(P, W, D)
    return (f, p_out, X, i)


def print_result(x):
    print("Final permutation: ", x[1])
    print("Final cost value: ", x[0])
    print("Final optimized x vector: ", x[2])
    print("Final number of iterations: ", x[3])
