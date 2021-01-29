import os
import numpy as np
from numpy.linalg import norm
from utils import *
from solvers import solve_qap
from load_test import load

"""
test: tuple of collection ("qaplib") and test ("chr12a")
solver: "sfw" or "tos"
random: if true, shift x0 by doubly stochastic random matrix
"""
def single_test(test, solver, stop_tol, random=False):
    W, D, best_f, best_P = load(test)
    X, P, t, iters = solve_qap(W, D, solver, random, stop_tol)
    return {
        "X": X, "P": P, "time": t, "iters": iters, "f_X": f_no_sparse(X, W, D), "f_P": f_no_sparse(P, W, D),
        "best_f": best_f, "best_P": best_P
    }

### Test suite building blocks ###

def test_k_times(test, solver, k, stop_tol):
    results = []
    for _ in range(k):
        results.append(single_test(test, solver, stop_tol, random=True))
    return results

# taking whichever result object has min f_P, though we're summing time across all
# result objects
def min_of_k(test, solver, k, stop_tol):
    results = test_k_times(test, solver, k, stop_tol)
    min_result = min(results, key = lambda t: t["f_P"])
    sum_time = sum([t["time"] for t in results])
    min_result["time"] = sum_time
    min_result["iters_min_of_k"] = min([t["iters"] for t in results])
    min_result["iters_mean_of_k"] = sum([t["iters"] for t in results])/len(results)
    return min_result

def min_of_k_n_times(test, solver, k, n, stop_tol = 1e-4):
    results = []
    for _ in range(n):
        results.append(min_of_k(test, solver, k, stop_tol))
    return {k: [dic[k] for dic in results] for k in results[0]}

# give mean and variance of cost and time values
def summarize(results):
    def mean_stdev(i):
        return {"mean": np.mean(results[i]), "stdev": np.std(results[i])}
    def mean_min_max(i):
        l = results[i]
        return {"mean": np.mean(l), "min": np.min(l), "max": np.max(l)}
    def norm_diff(l1,l2):
        if np.shape(l1[0]) == () or np.shape(l2[0]) == ():
            return None
        else:
            return np.mean([norm(l1[i]-l2[i]) for i in range(len(l1))])
    def birkhoff_inf(X):
        return norm(X - X.clip(0))/norm(X)
    return {
        "cost": mean_stdev("f_P"),
        "time": mean_stdev("time"),
        "iters": {"best_mean": np.mean(results["iters"]),
                  "min_of_k_mean": np.mean(results["iters_min_of_k"]),
                  "mean_of_k_mean": np.mean(results["iters_mean_of_k"]),
                  "max_reached(1e5)": max(results["iters"]) == 1e5
        },
        "cost_pre_rounded": np.mean(results["f_X"]),
        "cost_best_known": results["best_f"][0],
        "norm_diff": {"best_prerounded": norm_diff(results["best_P"], results["X"]),
                      "best_rounded" : norm_diff(results["best_P"], results["P"]),
                      "prerounded_rounded": norm_diff(results["X"], results["P"])
        },
        "birkhoff_infeasibility": np.mean([birkhoff_inf(x) for x in results["X"]])
    }

# get list of tests from collection name
def collection(collection):
    collections = os.listdir("data")
    assert collection in collections
    test_names = [x[:-4] for x in os.listdir(f"data/{collection}/input")]
    return [(collection, test_name) for test_name in test_names]
