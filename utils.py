import numpy as np
import scipy
from scipy import sparse
import lap

def n_(A):
    m, n = A.shape
    if m != n:
        raise "non-square input matrix"
    return n

def stack(P, Q, n):
    return np.concatenate(( P.reshape(n*n), Q.reshape(n*n)))

def unstack(x, n):
    P = x[0:n*n].reshape(n, n)
    Q = x[n*n:2*n*n].reshape(n, n)
    return (P,Q)

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

def fun(x, A, B):
    n = n_(A)
    P, Q = unstack(x, n)
    return (np.sum(P.dot(A).dot(P.T)*B))

def fungrad(x, A, B):
    n = n_(A)
    P, Q = unstack(x, n)
    f0 = fun(x,A,B)
    g = B.dot(Q).dot(A.T).reshape(n*n)
    g = np.concatenate((g, g))
    return (f0, g)

# TODO: make sure there aren't any major differences between the python lapjv
# function and the one they code up
def lapjv(C, resolution):
    cost, x, y = lap.lapjv(C)
    return (x, cost)

def maxassign_linprog(C):
    n = n_(C)
    if m>n:
        raise ValueError("matrix cannot have more rows than columns")
    A = np.concatenate((np.kron(np.eye(n), np.ones((1,n))), np.kron(np.ones((1,n)), np.eye(n))))
    A = A[:-1,]
    b = np.ones((2*n-1,1))
    c = -C.reshape(n*2)
    res = linprog(c, A_eq=A, b_eq=b, bounds=(0,1), options={"disp": True})
    X = res.x.reshape(n,n)
    x = X.T
    temp,p = np.max(x, axis = 1), np.argmax(x, axis = 1)
    w = -c.dot(res.x)
    p = p[1:n]
    return (p, w, x)

def perm2mat(p):
    n = len(p)
    P = np.zeros((n,n))
    for i in range(n):
        P[i, p[i]] = 1
    return P

def assign(A, munk = True):
    if munk:
        p, w = lapjv(-A.T, 0.01)
        x = perm2mat(p).T
        return p, -w, x
    else:
        p, w, x = maxassign_linprog(A.T)
        return (p.T, w, x)

def dsproj(x, g, n):
    P, Q = unstack(x, n)
    gP, gQ = unstack(g, n)
    q, wq, wQ = assign(-gQ)
    wP = wQ
    w = stack(wP, wQ, n)
    d = w-x
    return (d, q)


