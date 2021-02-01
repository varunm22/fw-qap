import math
import time
from utils import *
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csr_matrix

def iter_info(info, i, t0, feasibility, stationarity):
    info["iter"].append(i)
    info["time-elapsed"].append(time.time() - t0)
    info["feasibility"].append(feasibility)
    info["stationarity"].append(stationarity)

def sfw(W, D, X0, stop_tol = 1e-4, i_max = 1e4):
    n = n_(W)
    X = X0
    i = 0
    stop = 0
    info = {"iter": [], "time-elapsed": [], "feasibility": [], "stationarity": []}
    t0 = time.time()
    while (i < i_max and stop == 0):
        f0, g = f_(X, W, D), g_(X, W, D)
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

        if i == 0 or math.log(i, 2).is_integer():
            iter_info(info, i, t0, 0, gap) 

        i += 1

    if i == i_max:
        print("sfw no converge")

    return X, i, info

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


def tos_v2_proj1(X):
    X_ = X.clip(0)
    return X_/(X_.sum(axis=0)[np.newaxis,:])

def tos_v2_proj2(X):
    X_ = X.clip(0)
    return X_/(X_.sum(axis=1)[:,np.newaxis])

def tos_(W, D, X0, stop_tol, i_max, proj1, proj2):
    n = n_(W)
    # these can be tuned
    if sp.issparse(W):
        nW = sp_linalg.svds(W, 1, return_singular_vectors=False)
    else:
        nW = norm(W, 2)
    if sp.issparse(D):
        nD = sp_linalg.svds(D, 1, return_singular_vectors=False)
    else:
        nD = norm(D, 2)
    L = nW * nD
    L = L if L != 0 else 1e-6
    s = 1/L
    # try 4/3L and 1.99/L as well
    l = 1

    X = X0.astype('float32')
    Y = X
    i = 0
    stop = 0
    info = {"iter": [], "time-elapsed": [], "feasibility": [], "stationarity": []}
    t0 = time.time()

    ## an adaptive variant (heuristic) -- you need to uncomment the part below as well
    # SS = 0;
    # DD = math.sqrt(n)

    while (i < i_max and stop == 0):
        # X_old = np.copy(X)
        Z = proj1(Y)
        d = g_(Z, W, D)

        ## an adaptive variant (heuristic) -- you need to uncomment the part above as well
        # SS = SS + norm(d)**2
        # s = DD / (2*math.sqrt(SS))

        X = proj2(2*Z - Y - s*d)
        Y = Y + l*(X-Z)
        err = (norm(X-Z)/max(1,norm(X)))
        if err < stop_tol and i > 10:
            # print("tos", i)
            # print(f_(X, W, D))
            stop = 1

        if i == 0 or math.log(i, 2).is_integer():
            _, _, q = lap.lapjv(d.T)
            Q = perm2mat(q)
            d_ = Q-X
            gap = -np.sum(d*d_)
            iter_info(info, i, t0, norm(X-Z), gap)

        i += 1

    if i == i_max:
        print("tos no converge")

    return X, i, info

def tos(W, D, X0, stop_tol = 1e-4, i_max = 1e4):
    return tos_(W, D, X0, stop_tol, i_max, tos_proj1, tos_proj2)

def tos_v2(W, D, X0, stop_tol = 1e-4, i_max = 1e4):
    return tos_(W, D, X0, stop_tol, i_max, tos_v2_proj1, tos_v2_proj2)

def project_tos_to_birkhoff(X_in, maxit=int(1e3)):
    Y = np.copy(X_in)
    s = 1
    l = 1
    for i in range(maxit):
        Z = tos_proj1(Y)
        X = tos_proj2(2*Z - Y - s*(Z-X_in))
        Y += l*(X-Z)
    return X

def tos_bp_project(W, D, X0, stop_tol = 1e-4, i_max = 1e4):
    X, iters = tos(W, D, X0, i_max, stop_tol)
    return project_tos_to_birkhoff(X), iters

def solve_qap(W, D, solver, random, stop_tol):
    n = n_(W)
    if random:
        # rand_dir = np.random.normal(np.zeros(n**2), np.ones(n**2), n**2)
        # rand_dir = np.abs(rand_dir).reshape((n,n))
        # X0 = sink(rand_dir, 10)
        X0 = np.random.normal(size=W.shape)/n
        X0 = project_tos_to_birkhoff(X0)
    else:
        X0 = np.ones((n, n))/n

    X0 = X0.astype('float32')

# lam = 0.5
# X0 = (1-lam)*X0 + lam*sink(np.random.random(W.shape), 10)

    t0 = time.time()
    if solver == "sfw":
        X, iters, info = sfw(W, D, X0, stop_tol, i_max = 1000000)
    elif solver == "tos":
        X, iters, info = tos(W, D, X0, stop_tol, i_max = 100000)
    elif solver == "tos_v2":
        X, iters, info = tos_v2(W, D, X0, stop_tol, i_max = 100000)
    elif solver == "tos-bp-project":
        X, iters, info = tos_bp_project(W, D, X0, stop_tol, i_max = 100000)
    else:
        raise "not valid"
    t = time.time()-t0

    _, p_real, _ = lap.lapjv(-X)
    P = perm2mat(p_real)
    return X, P, t, iters, info
