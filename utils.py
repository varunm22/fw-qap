import numpy as np
import scipy
from scipy import sparse as sp
import scipy.sparse.linalg as sp_linalg
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
    return np.sum(X.T @ W @ X * D)

def g_(X, W, D):
    if is_symmetric(W) and is_symmetric(D):
        # reduces computation for symmetric matrices
        return W.dot(X).dot(D)  # W @ X @ D
    else:
        return (W.T.dot(X).dot(D.T) + W.dot(X).dot(D)) / 2  # (W.T @ X @ D.T + W @ X @ D)/2

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


def is_symmetric(X, tol=1e-8):
    if sp.issparse(X):
        if sp_linalg.norm(X - X.T, 'fro') < tol:
            return True
        else:
            return False
        # # Code below might be more efficient, we need to try.
        # X = sp.coo_matrix(X)
        # r, c, v = X.row, X.col, X.data
        # tril_no_diag = r > c
        # triu_no_diag = c > r
        # if triu_no_diag.sum() != tril_no_diag.sum():
        #     return False
        # rl = r[tril_no_diag]
        # cl = c[tril_no_diag]
        # vl = v[tril_no_diag]
        # ru = r[triu_no_diag]
        # cu = c[triu_no_diag]
        # vu = v[triu_no_diag]
        # sortl = np.lexsort((cl, rl))
        # sortu = np.lexsort((ru, cu))
        # vl = vl[sortl]
        # vu = vu[sortu]
        # return np.allclose(vl, vu)
    else:
        if np.linalg.norm(X - X.T, 'fro') < tol:
            return True
        else:
            return False

