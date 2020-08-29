import math
from utils import *
import numpy as np

def sfw(W, D, i_max = 30, x0=None, stop_tol = 0.0001):
    n = n_(W)
    if x0 == None:
        x0 = np.ones(n*n)/n
    elif x0 == -1: # random start near center of space
        X = np.ones(n)/n
        lam = 0.5
        x0 = (1-lam)*X + lam*sink(np.random.random(W.shape), 10)
    elif x0.shape == (W.shape[0],): # permutation provided as 1d array
        x0 = perm2mat(x0).T
    else: # actual permutation array provided
        pass

    x = x0.reshape(n*n)
    myp = []
    i = 0
    stop = 0
    myps = np.zeros((math.ceil(i_max), n))
    while (i < i_max and stop == 0):
        f0, g = f_(x, W, D), g_(x, W, D)
        _, q, _ = lap.lapjv(g.reshape((n,n)))
        Q = perm2mat(q).reshape(n*n)
        d = Q-x
        _f0_new, alpha = line_search(x, d, g, W, D)
        stop_norm = np.linalg.norm(alpha*Q)
        if stop_norm < np.spacing(1):
            stop = 1
        i += 1
        x = x + alpha*d

    # TODO: should this be transpose?
    _cost, p_out, y = lap.lapjv(-x.reshape((n,n)))
    P = perm2mat(p_out).T
    f = f_(P.reshape(n*n), W, D)
    return (f, p_out, x, i)


def print_result(x):
    print("Final permutation: ", x[1])
    print("Final cost value: ", x[0])
    print("Final optimized x vector: ", x[2])
    print("Final number of iterations: ", x[3])
