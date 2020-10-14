import os
import time
import numpy as np
from numpy.linalg import norm
from utils import *
from solvers import sfw, tos

def solve_qap(W, D, solver, random):
    n = n_(W)
    X0 = np.ones((n, n))/n
    if random:
        lam = 0.5
        X0 = (1-lam)*X0 + lam*sink(np.random.random(W.shape), 10)

    t0 = time.time()
    if solver == "sfw":
        X, iters = sfw( W, D, X0, i_max = 100000)
    elif solver == "tos":
        X, iters = tos(W, D, X0, i_max = 100000)
    else:
        raise "not valid"
    t = time.time()-t0

    _, p_real, _ = lap.lapjv(-X)
    P = perm2mat(p_real)
    return X, P, t, iters

"""
test: tuple of collection ("qaplib") and test ("chr12a")
solver: "sfw" or "tos"
random: if true, shift x0 by doubly stochastic random matrix
"""
def single_test(test, solver, random=False):
    collection, test_name = test
    def process_file(in_or_out, ext):
        path = f"data/{collection}/{in_or_out}/{test_name}.{ext}"
        file_lines = open(path, 'r').readlines()
        file_contents = " ".join(file_lines).replace("\n"," ").split()
        return [int(elt) for elt in file_contents]

    def generate_in():
        elts = process_file("input", "dat")
        n = elts.pop(0)
        assert len(elts) == 2*n**2
        return np.array(elts[:n**2]).reshape((n,n)), np.array(elts[n**2:]).reshape((n,n))

    def generate_out():
        elts = process_file("output", "sln")
        n = elts.pop(0)
        f = elts.pop(0)
        assert len(elts) == n
        best_P = perm2mat(np.array(elts)-1)
        return(f, best_P)

    W, D = generate_in()
    X, P, t, iters = solve_qap(W, D, solver, random)
    best_f, best_P = generate_out()
    # TODO: This should eventually be asserted, figure out why some aren't same
    if (best_f!=f_(best_P, W, D)):
        print(best_f, f_(best_P, W, D))
    return {
        "X": X, "P": P, "time": t, "iters": iters, "f_X": f_(X, W, D), "f_P": f_(P, W, D),
        "best_f": best_f, "best_P": best_P
    }

### Test suite building blocks ###

def test_k_times(test, solver, k):
    results = []
    for _ in range(k):
        results.append(single_test(test, solver, random=True))
    return results

# taking whichever result object has min f_P, though we're summing time across all
# result objects
def min_of_k(test, solver, k):
    results = test_k_times(test, solver, k)
    min_result = min(results, key = lambda t: t["f_P"])
    sum_time = sum([t["time"] for t in results])
    min_result["time"] = sum_time
    return min_result

def min_of_k_n_times(test, solver, k, n):
    results = []
    for _ in range(n):
        results.append(min_of_k(test, solver, k))
    return {k: [dic[k] for dic in results] for k in results[0]}

# give mean and variance of cost and time values
# can be used for both test_k_times and min_of_k_n_times results
def summarize(results):
    def mean_stdev(i):
        return {"mean": np.mean(results[i]), "stdev": np.std(results[i])}
    def mean_min_max(i):
        l = results[i]
        return {"mean": np.mean(l), "min": np.min(l), "max": np.max(l)}
    def norm_diff(l1,l2):
        return np.mean([norm(l1[i]-l2[i]) for i in range(len(l1))])
    return {
        "cost": mean_stdev("f_P"),
        "time": mean_stdev("time"),
        "iters": mean_min_max("iters"),
        "cost_pre_rounded": np.mean(results["f_X"]),
        "cost_best_known": results["best_f"][0],
        "norm_diff": {"best_prerounded": norm_diff(results["best_P"], results["X"]),
                      "best_rounded" : norm_diff(results["best_P"], results["P"]),
                      "prerounded_rounded": norm_diff(results["X"], results["P"])
        }
    }

# get list of tests from collection name
def collection(collection):
    collections = os.listdir("data")
    assert collection in collections
    test_names = [x[:-4] for x in os.listdir(f"data/{collection}/input")]
    return [(collection, test_name) for test_name in test_names]

# print(summarize(min_of_k_n_times(("qaplib", "chr12a"), "sfw", 1, 2)))
# print(summarize(min_of_k_n_times(("qaplib", "chr12a"), "tos", 1, 2)))
