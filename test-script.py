import sys
import numpy as np
from sfw import sfw
import utils

num_tests = 7

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

expected_costs = [10, 395, 314, 313, 470, 904, 1160]
def repeated_test(num, repetitions):
    expected_cost = expected_costs[int(num)-1]
    min_cost = None
    tests_passed = 0
    for i in range(repetitions):
        p, f = single_test(num, verbose=0, random=True)
        if min_cost == None:
            min_cost = f
        elif f < min_cost:
            min_cost = f
        if f == expected_cost:
            tests_passed += 1
            # print(tests_passed, i, p, f)
    if tests_passed == 0:
        print(f"No tests passed for expected {expected_cost}, min cost found was {min_cost}")
    else:
        print(f"Passed {tests_passed} out of {repetitions} test runs")

def test_suite(arg):
    if arg == "all":
        for i in range(num_tests):
            single_test(i+1, verbose=1)
    else:
        try:
            num = int(arg)
        except ValueError:
            print(f"invalid input, please send 'all' or a number from 1 to {num_tests}")
            return

        single_test(num, verbose=2)


# test_suite(sys.argv[1])
for i in range(num_tests):
    print(i+1)
    repeated_test(i+1, 1000)
