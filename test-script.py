import sys
import numpy as np
import sfw
import utils

# TODO: update doc for total number of tests, update README

num_tests = 7

def single_test(num, verbose=False):
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

    A, B = generate_in()
    f_real, p_real, _, _ = sfw.sfw(A, B, i_max = 50) # ignored args are final x and num iters
    p, f = generate_out()
    result = np.array_equal(p, p_real) and f == f_real
    result_str = "PASS" if result else "FAIL"
    print(f"Test number {num} result: {result_str}")
    if not result or verbose:
        print("EXPECTED RESULTS")
        print("p: ", p)
        print("f: ", f)
        print("OBTAINED RESULTS")
        print("p: ", p_real)
        print("f: ", f_real)

def test_suite(arg):
    if arg == "all":
        for i in range(num_tests):
            single_test(i+1)
    else:
        try:
            num = int(arg)
            single_test(num, verbose=True)
        except ValueError:
            print(f"invalid input, please send 'all' or a number from 1 to {num_tests}")


test_suite(sys.argv[1])
