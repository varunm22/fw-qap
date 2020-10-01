import sys
from functools import partial
import numpy as np
from sfw import sfw
from tos import tos
import utils
import matplotlib
import matplotlib.pyplot as plt

num_tests = 7
optimal_costs = [10, 395, 314, 313, 470, 904, 1160]

"""
verbose:
    0: don't print
    1: print expected vs obtained if non optimal
    2: print expected vs obtained
random: if true, shift x0 by doubly stochastic random matrix
"""
def single_test(num, verbose=0, random=False):
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
    if random:
        x0 = -1
    else:
        x0 = None
    f_real, p_real, _, _ = sfw( W, D, i_max = 50, x0=x0) # ignored args are final x and num iters
    # f_real, p_real, _, _ = tos( W, D, i_max = 50, X0=-1) # ignored args are final x and num iters
    p, f = generate_out()
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
    return (p_real, f_real)

### Test suite building blocks ###
#TODO: make these all gather timings as well

def test_k_times(num, k):
    costs = []
    for _ in range(k):
        costs.append(single_test(num, random=True))
    return costs

def min_of_k(num, k):
    costs = test_k_times(num, k)
    return (min(costs, key = lambda t: t[1]))

def min_of_k_n_times(num, k, n):
    costs = []
    for _ in range(n):
        costs.append(min_of_k(num, k))
    return costs

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


def pprint(l):
    num_min = 0
    avg = 0
    for i in l:
        if i[1] == 314:
            num_min += 1
        avg += i[1]
        print(i)
    print("num min:", num_min, "/", len(l))
    print("avg:", avg/len(l))

# print(for_all_tests(partial(min_of_k_n_times, k=20, n=1)))
pprint(test_k_times(6, 500))
