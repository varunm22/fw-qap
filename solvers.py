import math
import time
from utils import *
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix

def sfw(W, D, W_s, D_s, X0, stop_tol = 1e-4, i_max = 1e4):
    n = n_(W)
    X = X0
    i = 0
    stop = 0
    while (i < i_max and stop == 0):
        f0, g = f_(X, W, D, W_s, D_s), g_(X, W, D, W_s, D_s)
        _, _, q = lap.lapjv(g.T)
        Q = perm2mat(q)
        d = Q-X
        gap = -np.sum(g*d)
        if f0 < np.spacing(1) or (gap/f0 < stop_tol and norm(X) >= 1):
            stop = 1

        # this is exact step size, can also use approximation of:
        # eta = 2/(i+1)
        eta = line_search(X, d, g, W, D)
        if eta == 0:
            stop = 1

        X = X + eta*d
        if i%10 == 0:
            print("sfw", i, f0)
        i += 1

    if i == i_max:
        print("sfw no converge")

    return X, i

def tos_proj1(X):
    return X.clip(0)

def tos_proj2(X):
    n = X.shape[0]
    sX1 = np.sum(X, axis=0, keepdims=True)
    sX2 = np.sum(X, axis=1, keepdims=True)
    sXa = np.sum(sX1)
    return X + (1/n + (sXa/n**2)) - ((1/n)*sX2) - ((1/n)*sX1)

    # n = X.shape[0]
    # one = np.ones((n,1))
    # Z = (1/n + np.sum(X)/n**2)*np.eye(n) - (1/n)*X
    # return X + Z.dot(one).dot(one.T) - 1/n*one.dot(one.T).dot(X)

def tos(W, D, W_s, D_s, X0, stop_tol = 1e-4, i_max = 1e4):
    stop_tol = 1e-8
    n = n_(W)
    # these can be tuned
    L = norm(W)*norm(D)
    L = L if L != 0 else 1e-6
    print("L", L)
    s = 1/L
    l = 1

    X = X0
    Y = X
    i = 0
    stop = 0
    while (i < i_max and stop == 0):
        X_old = np.copy(X)
        Z = tos_proj1(Y)
        d = g_(Z, W, D, W_s, D_s)
        X = tos_proj2(2*Z - Y - s*d)
        Y += l*(X-Z)
        if (norm(X-X_old)/max(1,norm(X))) < stop_tol:
            stop = 1

        if i%10 == 0:
            print("tos", i)
        if i%100 == 0:
            print(f_(X, W, D, W_s, D_s))
        i += 1

    if i == i_max:
        print("tos no converge")

    return X, i

def project_tos_to_birkhoff(X_in):
    Y = np.copy(X_in)
    s = 1
    l = 1
    for i in range(int(1e3)):
        Z = tos_proj1(Y)
        X = tos_proj2(2*Z - Y - s*(Z-X_in))
        Y += l*(X-Z)
    return X

def tos_bp_project(W, D, W_s, D_s, X0, stop_tol = 1e-4, i_max = 1e4):
    X, iters = tos(W, D, X0, i_max, stop_tol)
    return project_tos_to_birkhoff(X), iters

def solve_qap(W, D, solver, random, stop_tol):
    n = n_(W)
    X0 = np.ones((n, n))/n
    if random:
        lam = 0.5
        X0 = (1-lam)*X0 + lam*sink(np.random.random(W.shape), 10)

    if np.sum(W != 0) / (W.shape[0]*W.shape[1]) < 0.25:
        W_s = csr_matrix(W)
        print("replaced W")
    else:
        W_s = W
    if np.sum(D != 0) / (D.shape[0]*D.shape[1]) < 0.25:
        D_s = csr_matrix(D)
        print("replaced D")
    else:
        D_s = D

    t0 = time.time()
    if solver == "sfw":
        X, iters = sfw(W, D, W_s, D_s, X0, stop_tol, i_max = 100000)
    elif solver == "tos":
        X, iters = tos(W, D, W_s, D_s, X0, stop_tol, i_max = 100000)
    elif solver == "tos-bp-project":
        X, iters = tos_bp_project(W, D, W_s, D_s, X0, stop_tol, i_max = 100000)
    else:
        raise "not valid"
    t = time.time()-t0

    _, p_real, _ = lap.lapjv(-X)
    P = perm2mat(p_real)
    return X, P, t, iters
