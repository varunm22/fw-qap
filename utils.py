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
    return np.sum(X.dot(W).dot(X.T)*D)/2

def g_(X, W, D):
    # return (W.dot(P).dot(D.T) + W.T.dot(P).dot(D)).reshape(n*n)
    # below is exactly half of above when W,D are symmetric, which should be always
    return W.dot(X).dot(D.T)

# TODO: make sure there aren't any major differences between the python lapjv
# function and the one they code up
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
    b = np.sum(g*d)
    c = f_(X,W,D)
    f_vertex = f_(X+d,W,D)
    a = f_vertex - b - c
    if (abs(a) < np.spacing(1)): # function is linear
        alpha = 1.
    else:
        alpha = min(1,max(-b/(2*a),0))
    f_alpha = f_(X+alpha*d,W,D)

    if (f_alpha>c):
        alpha = 0
        f_alpha = c

    if (f_alpha > f_vertex):
        alpha = 1
        f_alpha = f_vertex
        f0new = f_alpha

    X_t = X + alpha*d
    f0new = f_(X_t, W, D)

    return (f0new, alpha)
