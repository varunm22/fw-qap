import numpy as np
import scipy
from scipy import sparse
import lap

def stack(P, Q, m, n):
    return np.concatenate(( P.reshape(m*m), Q.reshape(n*n)))

def unstack(x, m, n):
    P = x[0:m*m].reshape(m, m)
    Q = x[m*m:m*m+n*n].reshape(n, n)
    return (P,Q)

def stoch(A, dim = 1):
    m, n = A.shape
    if dim == 1:
        s = np.sum(A, 1)
        if any(s == 0):
            print("Zero sum found")
            s = np.maximum(s, np.finfo(float).tiny)
        else:
            A = scipy.sparse.spdiags(1/s,0,m,m) * A
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
   m, n = A.shape
   P, Q = unstack(x, m, n)
   return (np.sum(P.dot(A).dot(Q.T)*B))

def fungrad(x, A, B):
    m, n = A.shape
    P, Q = unstack(x, m, n)
    f0 = fun(x,A,B)
    g = B.dot(Q).dot(A.T).reshape(m*m)
    g = np.concatenate((g, B.T.dot(P).dot(A).reshape(n*n)))
    return (f0, g)

# TODO: make sure there aren't any major differences between the python lapjv
# function and the one they code up
def lapjv(C, resolution):
    cost, x, y = lap.lapjv(C)
    return (x, cost)

def maxassign_linprog(C):
    m, n = C.shape
    if m>n:
        raise ValueError("matrix cannot have more rows than columns")
    C = np.concatenate((C, np.zeros((n-m, n))))
    A = np.concatenate((np.kron(np.eye(n), np.ones((1,n))), np.kron(np.ones((1,n)), np.eye(n))))
    A = A[:-1,]
    b = np.ones((2*n-1,1))
    c = -C.reshape(n*2)
    res = linprog(c, A_eq=A, b_eq=b, bounds=(0,1), options={"disp": True})
    X = res.x.reshape(n,n)
    x = X.T
    temp,p = np.max(x, axis = 1), np.argmax(x, axis = 1)
    w = -c.dot(res.x)
    p = p[1:m]
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

def dsproj(x, g, m, n):
    P, Q = unstack(x, m, n)
    gP, gQ = unstack(g, m, n)
    q, wq, wQ = assign(-gQ)
    wP = wQ
    w = stack(wP, wQ, m, n)
    d = w-x
    return (d, q)


