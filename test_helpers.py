import sys
import time
from functools import partial
import numpy as np
from sfw import sfw
from tos import tos
from utils import *
import matplotlib
import matplotlib.pyplot as plt

num_tests = 7
optimal_costs = [10, 395, 314, 313, 470, 904, 1160]

def solve_qap(W, D, solver, random):
    n = n_(W)
    X0 = np.ones((n, n))/n
    if random:
        lam = 0.5
        X0 = (1-lam)*X0 + lam*sink(np.random.random(W.shape), 10)

    t0 = time.time()
    if solver == "sfw":
        f_real, p_real, _, _ = sfw( W, D, X0, i_max = 100)
    elif solver == "tos":
        f_real, p_real, _, _ = tos(W, D, X0, i_max = 1000)
    else:
        raise "not valid"
    t = time.time()-t0
    return f_real, p_real, t

"""
verbose:
    0: don't print
    1: print expected vs obtained if non optimal
    2: print expected vs obtained
random: if true, shift x0 by doubly stochastic random matrix
"""
def single_test(num, solver, verbose=0, random=False):
    def process_file(suffix):
        file_lines = open("test-files/" + str(num) + "-" + suffix, 'r').readlines()
        return [[int(elt) for elt in line.strip().split(' ')] for line in file_lines]

    def generate_in():
        elts = process_file("in")
        n = len(elts[0])
        return(np.array(elts[:n]), np.array(elts[n:]))

    def generate_out():
        elts = process_file("out")
        return(np.array(elts[0]), elts[1][0])

    W, D = generate_in()
    p, f = generate_out()
    f_real, p_real, t = solve_qap(W, D, solver, random)
    if (verbose != 0):
        result = np.array_equal(p, p_real) and f == f_real
        result_str = "PASS" if result else "FAIL"
        print(f"Test number {num} result: {result_str}")
    if (verbose == 2 or (verbose == 1 and not result)):
        print("EXPECTED RESULTS")
        print("p: ", p)
        print("f: ", f)
        print("OBTAINED RESULTS")
        print("p: ", p_real)
        print("f: ", f_real)
    return (p_real, f_real, t)

### Test suite building blocks ###

def test_k_times(num, solver, k):
    results = []
    for _ in range(k):
        results.append(single_test(num, solver, random=True))
    return results

def min_of_k(num, solver, k):
    results = test_k_times(num, solver, k)
    perm, cost, _ = min(results, key = lambda t: t[1])
    t = sum([t[2] for t in results])
    return perm, cost, t

def min_of_k_n_times(num, solver, k, n):
    results = []
    for _ in range(n):
        results.append(min_of_k(num, solver, k))
    return results

# give mean and variance of cost and time values
# can be used for both test_k_times and min_of_k_n_times results
def summarize(results):
    def mean_and_stdev(i):
        l = [t[i] for t in results]
        return {"mean": np.mean(l), "stdev": np.std(l)}
    return { "cost": mean_and_stdev(1), "time": mean_and_stdev(2) }

# TODO: make this be for set of tests
def for_all_tests(f):
    results = {}
    for i in range(num_tests):
        results[i+1] = f(i+1)
    return results

### Data plotting functions ###

def histogram(num, costs):
    fig, ax = plt.subplots()
    ax.set_title("Costs for test number " + str(num) + " with min cost " + str(optimal_costs[num-1]))
    plt.hist(costs, 50)
    plt.show()

# print(for_all_tests(partial(min_of_k_n_times, k=20, n=1)))
# print(summarize(test_k_times(7, "sfw", 500)))
# print(summarize(test_k_times(7, "tos", 500)))
print(summarize(min_of_k_n_times(7, "sfw", 10, 100)))
print(summarize(min_of_k_n_times(7, "tos", 2, 100)))
