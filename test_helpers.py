import os
import sys
import time
import numpy as np
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
        X = sfw( W, D, X0, i_max = 10000)
    elif solver == "tos":
        X = tos(W, D, X0, i_max = 10000)
    else:
        raise "not valid"

    _, _, p_real = lap.lapjv(-X)
    P = perm2mat(p_real)
    f_real = f_(P, W, D)
    t = time.time()-t0
    return f_real, p_real, t

"""
verbose:
    0: don't print
    1: print expected vs obtained if non optimal
    2: print expected vs obtained
random: if true, shift x0 by doubly stochastic random matrix
"""
def single_test(test, solver, verbose=0, random=False):
    collection, test_name = test
    def process_file(in_or_out, ext):
        path = f"data/{collection}/{in_or_out}/{test_name}.{ext}"
        file_lines = open(path, 'r').readlines()
        file_lines = [l for l in file_lines if l != "\n"]
        return [[int(elt) for elt in line.strip().split()] for line in file_lines]

    def generate_in():
        elts = process_file("input", "dat")
        n = elts.pop(0)[0]
        return(np.array(elts[:n]), np.array(elts[n:]))

    def generate_out():
        elts = process_file("output", "sln")
        return(np.array(elts[1]), elts[1][1])

    W, D = generate_in()
    p, f = generate_out()
    f_real, p_real, t = solve_qap(W, D, solver, random)
    if (verbose != 0):
        result = np.array_equal(p, p_real) and f == f_real
        result_str = "PASS" if result else "FAIL"
        print(f"Test {test_name} from {collection} result: {result_str}")
    if (verbose == 2 or (verbose == 1 and not result)):
        print("EXPECTED RESULTS")
        print("p: ", p)
        print("f: ", f)
        print("OBTAINED RESULTS")
        print("p: ", p_real)
        print("f: ", f_real)
    return (p_real, f_real, t)

### Test suite building blocks ###

def test_k_times(test, solver, k):
    results = []
    for _ in range(k):
        results.append(single_test(test, solver, random=True))
    return results

def min_of_k(test, solver, k):
    results = test_k_times(test, solver, k)
    perm, cost, _ = min(results, key = lambda t: t[1])
    t = sum([t[2] for t in results])
    return perm, cost, t

def min_of_k_n_times(test, solver, k, n):
    results = []
    for _ in range(n):
        results.append(min_of_k(test, solver, k))
    return results

# give mean and variance of cost and time values
# can be used for both test_k_times and min_of_k_n_times results
def summarize(results):
    def mean_and_stdev(i):
        l = [t[i] for t in results]
        return {"mean": np.mean(l), "stdev": np.std(l)}
    return { "cost": mean_and_stdev(1), "time": mean_and_stdev(2) }

# get list of tests from collection name
def collection(collection):
    collections = os.listdir("data")
    assert collection in collections
    test_names = [x[:-4] for x in os.listdir(f"data/{collection}/input")]
    return [(collection, test_name) for test_name in test_names]

print(collection("qaplib"))
# print(summarize(min_of_k_n_times(("neos-guide", "7"), "sfw", 1, 100)))
# print(summarize(min_of_k_n_times(("neos-guide", "7"), "tos", 3, 10)))
# print(summarize(min_of_k_n_times(("qaplib", "chr12a"), "sfw", 3, 20)))
# print(summarize(min_of_k_n_times(("qaplib", "chr12a"), "tos", 3, 20)))
