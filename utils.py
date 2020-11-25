import numpy as np
import scipy
from scipy import sparse
import lap

def n_(W):
    m, n = W.shape
    if m != n:
        raise "non-square input matrix"
    return n

def stoch(A, dim = 1):
    n = n_(A)
    if dim == 1:
        s = np.sum(A, 1)
        if any(s == 0):
            print("Zero sum found")
            s = np.maximum(s, np.finfo(float).tiny)
        else:
            A = scipy.sparse.spdiags(1/s,0,n,n) * A
    else: # dim == 0
        s = np.sum(A, 0)
        if any(s == 0):
            print("Zero sum found")
            s = np.maximum(s, np.finfo(float).tiny)
        else:
            A = A * scipy.sparse.spdiags(1/s,0,n,n)
    return A

def sink(A, n):
    for i in range(n):
        A = stoch(A, 0)
        A = stoch(A, 1)
    return A

def f_(X, W, D):
    return np.sum(X.T.dot(W).dot(X)*D)

def g_(X, W, D):
    if np.all(W == W.T) and np.all(D == D.T):
        # reduces computation for symmetric matrices
        return W.dot(X).dot(D.T)
    else:
        return (W.dot(X).dot(D.T) + W.T.dot(X).dot(D))/2

def lapjv(C):
    cost, x, y = lap.lapjv(C)
    Q = np.zeros(C.shape)
    for i in range(n):
        Q[i, x[i]] = 1
    return Q

def perm2mat(p):
    n = len(p)
    P = np.zeros((n,n))
    for i in range(n):
        P[i, p[i]] = 1
    return P

def line_search(X, d, g, W, D):
    tA = np.sum(d*(W.dot(d).dot(D)))
    # print("tA: ", tA)
    if tA > 0:
        tB = np.sum(X*(W.dot(d).dot(D))) + np.sum(d*(W.dot(X).dot(D)))
        # print("tB: ", tB)
        eta = -tB/(2*tA)
        # print("eta raw: ", eta)
        eta = max(min(eta,1),0)
    else:
        eta = 1
    return eta
