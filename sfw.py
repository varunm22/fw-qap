import math
from utils import *
import numpy as np

def sfw(W, D, X0, i_max = 30, stop_tol = 0.0001):
    n = n_(W)
    X = X0
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
